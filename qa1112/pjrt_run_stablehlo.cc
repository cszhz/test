// 用 C++ 通过 PJRT C API 在 TPU 上执行 StableHLO（第三部分示例）。
//
// 流程：dlopen(libtpu.so) -> GetPjrtApi() -> 创建 client -> 编译 StableHLO bytecode
//       -> 上传输入 -> 执行 -> 取回输出。
// 输入产物由 aot_stablehlo_export.py 生成：
//   artifacts/model.stablehlo.mlirbc   StableHLO bytecode（含 Pallas 的 tpu_custom_call）
//   artifacts/compile_options.pb       xla.CompileOptionsProto（序列化）
// 示例模型的输入：bf16[4096,512] x2、bf16[512]、bf16[512,1024]；输出 bf16[4096,1024]。
//
// 编译：clang++-18（或 g++）-O2 -std=c++17 -I third_party pjrt_run_stablehlo.cc -ldl -o pjrt_run_stablehlo
// 运行（TPU 机器，且没有别的进程占用 TPU）：
//   ./pjrt_run_stablehlo /path/to/libtpu.so artifacts
// 只依赖 pjrt_c_api.h 这一个 C 头文件，不需要链接 XLA。

#include <dlfcn.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "xla/pjrt/c/pjrt_c_api.h"

namespace {

const PJRT_Api* api = nullptr;

void Check(PJRT_Error* error, const char* what) {
  if (error == nullptr) return;
  PJRT_Error_Message_Args msg{};
  msg.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
  msg.error = error;
  api->PJRT_Error_Message(&msg);
  std::fprintf(stderr, "%s failed: %.*s\n", what, static_cast<int>(msg.message_size),
               msg.message);
  PJRT_Error_Destroy_Args destroy{};
  destroy.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
  destroy.error = error;
  api->PJRT_Error_Destroy(&destroy);
  std::exit(1);
}

void AwaitAndDestroy(PJRT_Event* event, const char* what) {
  if (event == nullptr) return;
  PJRT_Event_Await_Args await{};
  await.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
  await.event = event;
  Check(api->PJRT_Event_Await(&await), what);
  PJRT_Event_Destroy_Args destroy{};
  destroy.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
  destroy.event = event;
  Check(api->PJRT_Event_Destroy(&destroy), "PJRT_Event_Destroy");
}

std::string ReadFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    std::fprintf(stderr, "cannot read %s\n", path.c_str());
    std::exit(1);
  }
  return std::string(std::istreambuf_iterator<char>(in), {});
}

// float -> bf16（截断到高 16 位，四舍五入到偶数）
uint16_t ToBf16(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, 4);
  bits += 0x7FFF + ((bits >> 16) & 1);
  return static_cast<uint16_t>(bits >> 16);
}

float FromBf16(uint16_t h) {
  uint32_t bits = static_cast<uint32_t>(h) << 16;
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

PJRT_Buffer* Upload(PJRT_Client* client, PJRT_Device* device,
                    const std::vector<uint16_t>& host, std::vector<int64_t> dims) {
  PJRT_Client_BufferFromHostBuffer_Args args{};
  args.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
  args.client = client;
  args.data = host.data();
  args.type = PJRT_Buffer_Type_BF16;
  args.dims = dims.data();
  args.num_dims = dims.size();
  args.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes;
  args.device = device;
  Check(api->PJRT_Client_BufferFromHostBuffer(&args), "BufferFromHostBuffer");
  AwaitAndDestroy(args.done_with_host_buffer, "upload");
  return args.buffer;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <libtpu.so> <artifacts_dir>\n", argv[0]);
    return 2;
  }
  const std::string dir = argv[2];

  // 1. 加载插件，拿到 PJRT C API 函数表
  void* lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (lib == nullptr) {
    std::fprintf(stderr, "dlopen: %s\n", dlerror());
    return 1;
  }
  using GetApiFn = const PJRT_Api* (*)();
  auto get_api = reinterpret_cast<GetApiFn>(dlsym(lib, "GetPjrtApi"));
  if (get_api == nullptr) {
    std::fprintf(stderr, "GetPjrtApi not found\n");
    return 1;
  }
  api = get_api();
  std::printf("PJRT C API %d.%d（头文件 %d.%d）\n", api->pjrt_api_version.major_version,
              api->pjrt_api_version.minor_version, PJRT_API_MAJOR, PJRT_API_MINOR);
  PJRT_Plugin_Initialize_Args init{};
  init.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE;
  Check(api->PJRT_Plugin_Initialize(&init), "PJRT_Plugin_Initialize");

  // 2. 创建 client，取第一个设备
  PJRT_Client_Create_Args create{};
  create.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
  Check(api->PJRT_Client_Create(&create), "PJRT_Client_Create");
  PJRT_Client* client = create.client;
  PJRT_Client_AddressableDevices_Args devices{};
  devices.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
  devices.client = client;
  Check(api->PJRT_Client_AddressableDevices(&devices), "AddressableDevices");
  std::printf("可用设备 %zu 个，使用第 0 个\n", devices.num_addressable_devices);
  PJRT_Device* device = devices.addressable_devices[0];

  // 3. 编译 StableHLO bytecode（format = "mlir"，文本或 bytecode 均可）
  std::string code = ReadFile(dir + "/model.stablehlo.mlirbc");
  std::string options = ReadFile(dir + "/compile_options.pb");
  PJRT_Program program{};
  program.struct_size = PJRT_Program_STRUCT_SIZE;
  program.code = code.data();
  program.code_size = code.size();
  program.format = "mlir";
  program.format_size = 4;
  PJRT_Client_Compile_Args compile{};
  compile.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
  compile.client = client;
  compile.program = &program;
  compile.compile_options = options.data();
  compile.compile_options_size = options.size();
  auto t0 = std::chrono::steady_clock::now();
  Check(api->PJRT_Client_Compile(&compile), "PJRT_Client_Compile");
  auto t1 = std::chrono::steady_clock::now();
  std::printf("编译 %.1f ms\n", std::chrono::duration<double, std::milli>(t1 - t0).count());
  PJRT_LoadedExecutable* exe = compile.executable;

  PJRT_LoadedExecutable_GetExecutable_Args get_exe{};
  get_exe.struct_size = PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE;
  get_exe.loaded_executable = exe;
  Check(api->PJRT_LoadedExecutable_GetExecutable(&get_exe), "GetExecutable");
  PJRT_Executable_NumOutputs_Args num_out{};
  num_out.struct_size = PJRT_Executable_NumOutputs_Args_STRUCT_SIZE;
  num_out.executable = get_exe.executable;
  Check(api->PJRT_Executable_NumOutputs(&num_out), "NumOutputs");

  // 4. 准备输入（和 Python 示例同样的 shape，随机数据）
  constexpr int64_t T = 4096, H = 512, O = 1024;
  std::srand(0);
  auto rnd = [](size_t n, float scale) {
    std::vector<uint16_t> v(n);
    for (auto& x : v) x = ToBf16(scale * (std::rand() / float(RAND_MAX) * 2 - 1));
    return v;
  };
  std::vector<uint16_t> x = rnd(T * H, 1.f), r = rnd(T * H, 1.f), w = rnd(H, 1.f),
                        wp = rnd(H * O, 1.f / std::sqrt(float(H)));
  PJRT_Buffer* inputs[4] = {Upload(client, device, x, {T, H}),
                            Upload(client, device, r, {T, H}),
                            Upload(client, device, w, {H}),
                            Upload(client, device, wp, {H, O})};

  // 5. 执行（单设备：argument_lists / output_lists 都只有一行）
  std::vector<PJRT_Buffer*> outputs(num_out.num_outputs, nullptr);
  PJRT_Buffer* const* arg_list = inputs;
  PJRT_Buffer** out_list = outputs.data();
  PJRT_Event* done = nullptr;
  PJRT_ExecuteOptions exec_options{};
  exec_options.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;
  PJRT_LoadedExecutable_Execute_Args exec{};
  exec.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
  exec.executable = exe;
  exec.options = &exec_options;
  exec.argument_lists = &arg_list;
  exec.num_devices = 1;
  exec.num_args = 4;
  exec.output_lists = &out_list;
  exec.device_complete_events = &done;
  t0 = std::chrono::steady_clock::now();
  Check(api->PJRT_LoadedExecutable_Execute(&exec), "Execute");
  AwaitAndDestroy(done, "execute");
  t1 = std::chrono::steady_clock::now();
  std::printf("执行 %.3f ms（第一次，含启动开销）\n",
              std::chrono::duration<double, std::milli>(t1 - t0).count());

  // 6. 取回输出
  std::vector<uint16_t> y(T * O);
  PJRT_Buffer_ToHostBuffer_Args to_host{};
  to_host.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
  to_host.src = outputs[0];
  to_host.dst = y.data();
  to_host.dst_size = y.size() * sizeof(uint16_t);
  Check(api->PJRT_Buffer_ToHostBuffer(&to_host), "ToHostBuffer");
  AwaitAndDestroy(to_host.event, "download");
  double sum = 0;
  for (uint16_t v : y) sum += std::fabs(FromBf16(v));
  std::printf("输出 bf16[%lld,%lld]，mean|y| = %.4f\n", static_cast<long long>(T),
              static_cast<long long>(O), sum / y.size());

  // 7. 释放
  for (PJRT_Buffer* b : inputs) {
    PJRT_Buffer_Destroy_Args d{};
    d.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
    d.buffer = b;
    Check(api->PJRT_Buffer_Destroy(&d), "Buffer_Destroy");
  }
  for (PJRT_Buffer* b : outputs) {
    PJRT_Buffer_Destroy_Args d{};
    d.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
    d.buffer = b;
    Check(api->PJRT_Buffer_Destroy(&d), "Buffer_Destroy");
  }
  PJRT_Executable_Destroy_Args de{};
  de.struct_size = PJRT_Executable_Destroy_Args_STRUCT_SIZE;
  de.executable = get_exe.executable;
  Check(api->PJRT_Executable_Destroy(&de), "Executable_Destroy");
  PJRT_LoadedExecutable_Destroy_Args dl{};
  dl.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
  dl.executable = exe;
  Check(api->PJRT_LoadedExecutable_Destroy(&dl), "LoadedExecutable_Destroy");
  PJRT_Client_Destroy_Args dc{};
  dc.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
  dc.client = client;
  Check(api->PJRT_Client_Destroy(&dc), "Client_Destroy");
  return 0;
}
