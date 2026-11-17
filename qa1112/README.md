# TPU 问答（2026-11-12）：host 内存乱序 gather、JIT / C++ / AOT 与 StableHLO

环境：JAX 0.11.2 / libtpu 0.0.50。本机用 `source env.sh`：设 `$PY`（venv `/disk/venvs/qa1112-jax`），并把进程限定在 2 号芯片
（本机其余芯片有服务在跑）；独占 TPU 机器时可不 source，自己设 `PY`。

| 文件 | 内容 |
|---|---|
| `host_gather_pallas.py` | 第二部分：host pinned 内存里的表，Pallas kernel 按乱序下标 DMA 取到 HBM |
| `aot_stablehlo_export.py` | 第三部分：JIT / AOT / `jax.export` 导出 StableHLO（示例模型带 Pallas kernel） |
| `pjrt_run_stablehlo.cc`、`pjrt_run_stablehlo` | 第三部分：C++ 经 PJRT C API 加载 libtpu.so 编译并执行导出的 StableHLO |
| `check_cpp_vs_jax.py` | 第三部分：逐位重建 C++ 的输入，在 JAX 里核对 C++ 的输出 |
| `runlogs/` | TPU 实测日志 |
| `pylib/` | 序列化 `Exported` 需要的 flatbuffers |
| `third_party/` | PJRT C API 头文件 |

## 第二部分：Embedding 大表在 host 内存、乱序下标 gather

| 文件 | 内容 |
|---|---|
| `host_gather_pallas.py` | 表放在 host pinned 内存（`memory_kind="pinned_host"`），Pallas TensorCore kernel 用 `pl.HOST` 内存空间直接 DMA 按乱序下标取数据到 HBM，不经过 CPU。按对齐块取（见下） |

**验证状态**：CPU 上导出 TPU 版 StableHLO 成功：表参数标为 `mhlo.memory_kind="pinned_host"`，直接作为 `tpu_custom_call` 的输入，
中间没有整表拷贝（TPU 上编译后的 HLO 也确认没有拷贝）。

TPU 上实测（2026-11-17，v6e，JAX 0.11.2 / libtpu 0.0.50）：
- 运行时下标的 host DMA 以对齐块为单位：host 数组行宽为 4 KiB 的整数倍，一次取对齐的 8 行（一个 `(8,128)` tile 的高度），
  即每块 8 行 × 4 KiB = 32 KiB。
- 0.5 GiB 的表（f32 `[*, 1024]`）里乱序取 2048 块，结果正确，**2.40 ms、约 28 GB/s、0.85 M 块/s**；块 64 KiB 时同样约 28 GB/s。
  v6e 每颗芯片的 PCIe 约 32 GB/s 单向（知识库 `wiki/concepts/pcie.md`）。

---

## 第三部分：JIT 运行 / C++ 接口 / AOT 与 StableHLO

| 文件 | 内容 |
|---|---|
| `aot_stablehlo_export.py` | JIT、AOT（`lower().compile()`）、`jax.export` 导出 StableHLO 并序列化/反序列化执行；可在 CPU 机器上导出 TPU 产物 |
| `pjrt_run_stablehlo.cc` | C++ 通过 PJRT C API（`dlopen libtpu.so` → `GetPjrtApi`）编译并执行导出的 StableHLO |
| `third_party/xla/pjrt/c/pjrt_c_api.h` | PJRT C API 头文件（v0.116，来自本机 tpu-sync 编译用的 XLA） |
| `artifacts/`（运行后生成，不入库） | `model.stablehlo.mlir`（文本）、`.mlirbc`（bytecode）、`model.jax_exported`（序列化 Exported）、`compile_options.pb` |
| `check_cpp_vs_jax.py` | 用 libc `srand(0)/rand()` 逐位重建 C++ 的输入，在 JAX 里跑同一模型，核对 C++ 输出 |

```bash
cd ~/temu && source env.sh
# 只导出（任何机器，不需要 TPU；序列化需要 flatbuffers，已装在 ./pylib）
PYTHONPATH=pylib JAX_PLATFORMS=cpu $PY aot_stablehlo_export.py --out artifacts
# TPU 上：Python 侧执行并对比 jit / AOT / 反序列化三种结果
PYTHONPATH=pylib $PY aot_stablehlo_export.py --out artifacts
# C++：编译（已编好 ./pjrt_run_stablehlo）并在 TPU 上运行（不能和别的进程同时占同一颗芯片）
clang++-18 -O2 -std=c++17 -I third_party pjrt_run_stablehlo.cc -ldl -o pjrt_run_stablehlo
./pjrt_run_stablehlo /disk/venvs/qa1112-jax/lib/python3.12/site-packages/libtpu/libtpu.so artifacts
# 核对 C++ 输出（mean|y| 应与 C++ 打印的一致）
$PY check_cpp_vs_jax.py
```

**验证状态**：CPU 上导出 TPU 版 StableHLO 成功（模块含 Pallas 的 `stablehlo.custom_call @tpu_custom_call`），
序列化/反序列化成功；C++ 示例 `-Wall` 编译通过。

TPU 上实测（2026-11-17，v6e 单芯片，日志在 `runlogs/`）：
- Python：jit / AOT（`lower().compile()`）/ 反序列化 `Exported.call` 三者输出逐位相同（max|err| = 0）。
- C++：PJRT C API 0.116，编译 StableHLO 约 380–395 ms（每次启动都要编），第一次执行 3–8 ms（含启动开销），
  输出 `bf16[4096,1024]`，mean|y| = 0.2622；`check_cpp_vs_jax.py` 用逐位相同的输入在 JAX 里跑同一模型，mean|y| 也是 0.2622。
  C++ 只打印 mean|y|，没有逐元素比对。

**要点**
- **JIT**：`jax.jit` 首次调用按输入 shape 编译，之后复用；`jax.config.update("jax_compilation_cache_dir", ...)` 开持久缓存，
  进程重启不重编。shape 变化会触发重编，动态长度要按桶 pad。
- **AOT**
  - `jit(f).lower(*ShapeDtypeStruct).compile()`：同进程里提前编好；`lowered.as_text()` 拿 StableHLO。
  - `jax.export.export(jit(f), platforms=["tpu"])(*shapes)`：得到 `Exported`，`.serialize()` 落盘（StableHLO + 调用约定 + 版本），
    另一进程 `jax.export.deserialize(...).call(...)`。JAX 保证一定时间窗口内（约 6 个月）新版本能加载旧产物。可以在 CPU 机器上导出 TPU 版本。
  - 想连编译都省掉：`jax.experimental.serialize_executable` 序列化编译好的可执行文件，但绑定具体 JAX/libtpu 版本和 TPU 拓扑。
- **StableHLO 与自定义 kernel**：Pallas kernel 在 StableHLO 里是 `tpu_custom_call`（内嵌序列化的 Mosaic kernel），在 `jax.export` 的
  "保证稳定"白名单里，所以带 Pallas kernel 的模型也能导出/部署。TPU 上**没有**类似 CUDA 的用户 C++ device kernel；自定义设备侧代码只能用
  Pallas/Mosaic 写。host 侧的自定义逻辑可用 `jax.pure_callback` / `io_callback`，或 `jax.ffi` 注册 CPU 上的 C++ 函数。
- **C++ 接口**：`libtpu.so` 导出 PJRT C API（`GetPjrtApi`）。C++ 程序只需 `pjrt_c_api.h` 一个头文件，`dlopen` 后创建 client、
  用 `format="mlir"` 编译 StableHLO（文本或 bytecode）、`BufferFromHostBuffer` 上传、`Execute`、`ToHostBuffer` 取回；不用链接 XLA。
  编译选项 `compile_options.pb`（`xla.CompileOptionsProto`）由 Python 侧生成；多设备/分片时在选项里设 replica/partition 数，
  并给 `Execute` 传每个设备一行参数。
- 其他部署路径：jax2tf → TF SavedModel → TF Serving（C++）；或直接用 vLLM / JetStream 这类现成的 TPU serving。
