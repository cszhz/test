# TPU 问答（2026-11-12）：host 内存乱序 gather、JIT / C++ / AOT 与 StableHLO

环境：JAX 0.11.2 / libtpu 0.0.50。本机用 `source env.sh`：设 `$PY`（venv `/disk/venvs/qa1112-jax`），并把进程限定在 2 号芯片
（本机其余芯片有服务在跑）；独占 TPU 机器时可不 source，自己设 `PY`。

| 文件 | 内容 |
|---|---|
| `host_gather_pallas.py` | 第二部分：host pinned 内存里的表，Pallas kernel 按乱序下标 DMA 取到 HBM |
| `aot_stablehlo_export.py` | 第三部分：JIT / AOT / `jax.export` 导出 StableHLO（示例模型带 Pallas kernel） |
| `pjrt_run_stablehlo.cc`、`pjrt_run_stablehlo` | 第三部分：C++ 经 PJRT C API 加载 libtpu.so 编译并执行导出的 StableHLO |
| `check_cpp_vs_jax.py` | 第三部分：逐位重建 C++ 的输入，在 JAX 里核对 C++ 的输出 |
| `artifacts/`、`artifacts_tpu/` | 在 CPU / TPU 上导出的产物 |
| `runlogs/` | TPU 实测日志 |
| `pylib/` | 序列化 `Exported` 需要的 flatbuffers |
| `third_party/` | PJRT C API 头文件 |

## 第二部分：Embedding 大表在 host 内存、乱序下标 gather

| 文件 | 内容 |
|---|---|
| `host_gather_pallas.py` | 表放在 host pinned 内存（`memory_kind="pinned_host"`），Pallas TensorCore kernel 用 `pl.HOST` 内存空间直接 DMA 按乱序下标取数据到 HBM，不经过 CPU。默认按对齐块取（见下），`--mode row` 复现逐行取的编译失败 |

**验证状态**：CPU 上导出 TPU 版 StableHLO 成功：表参数标为 `mhlo.memory_kind="pinned_host"`，直接作为 `tpu_custom_call` 的输入，
中间没有整表拷贝（TPU 上编译后的 HLO 也确认没有拷贝）。

TPU 上实测（2026-11-17，v6e，JAX 0.11.2 / libtpu 0.0.50；下面的限制只在 v6e 上验证过，TPU7x 和新版 libtpu 待测）：
- **逐行取（原始需求：表 `[4M, 256]` bf16、每行 512 B）编译不过。** 运行时下标的 host DMA 要求 host 数组的行宽是 4 KiB 的整数倍，
  且一次取对齐的 8 行（一个 `(8,128)` tile 的高度），即单次 DMA 最少 32 KiB；否则 Mosaic 报 tile 对齐错误，或 LLO 报
  `multiplier_in_bytes % word_size == 0 (512 4096)`。改成 `[rows, 1, dim]`（XLA 会把长度 1 的维挪到最外，tile 仍落在行上）、
  强制 `Layout((0,1,2))` 加小 tile `(1,128)`/`(2,128)`、行宽 2 KiB 等写法都试过，同样失败。静态偏移的 host DMA 没问题。
- **按对齐块取可以**：每块 8 行 × 4 KiB（f32 `[*, 1024]`）= 32 KiB，0.5 GiB 的表里乱序取 2048 块，结果正确，
  **2.40 ms、约 28 GB/s、0.85 M 块/s**；块 64 KiB 时同样约 28 GB/s。v6e 每颗芯片的 PCIe 约 32 GB/s 单向
  （知识库 `wiki/concepts/pcie.md`），已接近带宽上限。
- 折算到 512 B 的行：要逐行取，只能把目标行所在的整块（32 KiB）取回来，读放大 64 倍，有效带宽约 0.44 GB/s。

**结论**
1. **TPU 能直接 DMA pinned host 内存，但粒度很粗，做不成 CUDA 那样的逐行 zero-copy**：Pallas 对 TensorCore kernel 开放了
   `pl.HOST` 内存空间（lowering 为 `#tpu.memory_space<host>`），TPU 的 DMA 引擎经 PCIe 直接读 host 内存、不经过 CPU。
   但实测（见上）运行时下标的 DMA 最小单位是 8 行 × 4 KiB = 32 KiB，512 B 的行没法单独取。适用于"大块"冷数据
   （行宽 ≥ 4 KiB，或能按 8 行分组一起用的数据）；小行乱序 gather 仍要走 CPU gather 再整块上传，或把数据放进 HBM。
2. **SparseCore 不能直接读 host 内存**：Pallas 的 SparseCore lowering（`sc_lowering.py` / `sc_core.py`）没有 host 内存空间。
   SparseCore 的乱序 gather（`pl.Indirect`，见 `/zzlfs/ds41flash/tools/sc_big_table_gather_example.py`）只适用于 HBM 里的表，
   适合放 5–10% 的热点部分。
3. **推荐的分层做法**：热点行放 HBM（SparseCore 或 XLA gather）；冷行放 host（按目前的限制，512 B 小行用 CPU 多线程 gather 后整块上传，
   大块数据才用 TensorCore host-DMA kernel）；
   下标在 step 开始前就已知，可以提前一步把冷行 DMA 到 HBM 的 staging 区，和计算重叠，把 PCIe 延迟藏起来。
4. **现成 embedding 库的 CPU 瓶颈**：TPU embedding 栈（SparseCore embedding）在 host 上做输入预处理（按 SparseCore 分片、排序、去重），
   这是设计如此，不是可配置项。绕开的办法：用上面的 Pallas kernel 自己做 gather；或把预处理放到多线程 / 流水线里与设备计算重叠。
5. **词表会变**：现成库要求表大小固定。做法是预分配固定容量的表（留余量），维护 ID → 槽位的动态映射（host 上的哈希表，或设备上
   用 Pallas 实现），新 ID 分配空槽、满了按策略淘汰；容量不够时离线扩容。DeepSeek Engram 用 n-gram 哈希映射到固定大小的表，是同一思路。
6. **`SparseDenseMatmulOp` 闭源改不了**：不依赖它，用 Pallas 自己写：TensorCore 的 host-DMA gather（冷数据，受上面的块粒度限制）+ SparseCore 的
   indirect gather（热数据）+ XLA 做后续的组合（sum/mean pooling 等）。
7. 对照：vllm-torchtpu 里 DeepSeek-V4.1 的 Engram host 表现在是在 CPU 上用 numba gather 再拷到 TPU
   （`models/vllm/deepseek_v41/common/engram.py` 的 `gather_raw` + `host_gather.py`），也就是题目里想避免的"CPU 改造"方式；
   在逐行 host DMA 能用之前（行宽 ≥ 4 KiB 或 libtpu 放开限制），这条路还替换不了。

---

## 第三部分：JIT 运行 / C++ 接口 / AOT 与 StableHLO —— 都能接

| 文件 | 内容 |
|---|---|
| `aot_stablehlo_export.py` | JIT、AOT（`lower().compile()`）、`jax.export` 导出 StableHLO 并序列化/反序列化执行；可在 CPU 机器上导出 TPU 产物 |
| `pjrt_run_stablehlo.cc` | C++ 通过 PJRT C API（`dlopen libtpu.so` → `GetPjrtApi`）编译并执行导出的 StableHLO |
| `third_party/xla/pjrt/c/pjrt_c_api.h` | PJRT C API 头文件（v0.116，来自本机 tpu-sync 编译用的 XLA） |
| `artifacts/` | CPU 上导出的产物：`model.stablehlo.mlir`（文本）、`.mlirbc`（bytecode）、`model.jax_exported`（序列化 Exported）、`compile_options.pb` |
| `artifacts_tpu/` | TPU 上导出的同一套产物（C++ 实测用的这份） |
| `check_cpp_vs_jax.py` | 用 libc `srand(0)/rand()` 逐位重建 C++ 的输入，在 JAX 里跑同一模型，核对 C++ 输出 |

```bash
cd ~/temu && source env.sh
# 导出（任何机器；序列化需要 flatbuffers，已装在 ./pylib）
PYTHONPATH=pylib JAX_PLATFORMS=cpu $PY aot_stablehlo_export.py --out artifacts
# TPU 上：Python 侧执行并对比 jit / AOT / 反序列化三种结果
PYTHONPATH=pylib $PY aot_stablehlo_export.py --out artifacts_tpu
# C++：编译（已编好 ./pjrt_run_stablehlo）并在 TPU 上运行（不能和别的进程同时占同一颗芯片）
clang++-18 -O2 -std=c++17 -I third_party pjrt_run_stablehlo.cc -ldl -o pjrt_run_stablehlo
./pjrt_run_stablehlo /disk/venvs/qa1112-jax/lib/python3.12/site-packages/libtpu/libtpu.so artifacts_tpu
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
