# TPU 问答（2026-11-12）：host 内存乱序 gather、JIT / C++ / AOT 与 StableHLO

环境：JAX 0.11.2 / libtpu 0.0.50。

| 文件 | 内容 |
|---|---|
| `host_gather_pallas.py` | 第二部分：host pinned 内存里的表，Pallas kernel 按乱序下标 DMA 取到 HBM |
| `aot_stablehlo_export.py` | 第三部分：JIT / AOT / `jax.export` 导出 StableHLO（示例模型带 Pallas kernel） |
| `pjrt_run_stablehlo.cc`、`pjrt_run_stablehlo` | 第三部分：C++ 经 PJRT C API 加载 libtpu.so 编译并执行导出的 StableHLO |
| `check_cpp_vs_jax.py` | 第三部分：逐位重建 C++ 的输入，在 JAX 里核对 C++ 的输出 |
| `runlogs/` | TPU 实测日志 |
| `pylib/` | 序列化 `Exported` 需要的 flatbuffers |
| `third_party/` | PJRT C API 头文件 |
| `env.sh` | 只在多个进程分芯片用时需要：把进程限定在单颗芯片（2 号）并设 `$PY`；独占 TPU 机器不用 |

## 运行

在 `qa1112/` 目录下执行；C++ 运行时 TPU 不能被别的进程占用，以下命令按顺序跑。

```bash
git clone https://github.com/cszhz/test.git && cd test/qa1112

# 环境（flatbuffers 已在 ./pylib 里；装出来的 libtpu 不是 0.0.50 时另外指定 libtpu==0.0.50）
python3 -m venv .venv && .venv/bin/pip install "jax[tpu]==0.11.2" numpy
PY=.venv/bin/python
LIBTPU=$($PY -c "import libtpu, os; print(os.path.join(os.path.dirname(libtpu.__file__), 'libtpu.so'))")

# 第二部分：host 内存乱序 gather
PYTHONPATH=pylib $PY host_gather_pallas.py                                   # TPU 上执行并校验
PYTHONPATH=pylib JAX_PLATFORMS=cpu $PY host_gather_pallas.py --export-only   # 只导出检查，不需要 TPU

# 第三部分：JIT / AOT / StableHLO 导出 + C++ 执行
PYTHONPATH=pylib JAX_PLATFORMS=cpu $PY aot_stablehlo_export.py --out artifacts  # 只导出，不需要 TPU
PYTHONPATH=pylib $PY aot_stablehlo_export.py --out artifacts     # TPU 上导出，并对比 jit / AOT / 反序列化三种执行
clang++-18 -O2 -std=c++17 -I third_party pjrt_run_stablehlo.cc -ldl -o pjrt_run_stablehlo   # 可直接用已编好的二进制
./pjrt_run_stablehlo $LIBTPU artifacts                           # C++ 经 PJRT 在 TPU 上执行
$PY check_cpp_vs_jax.py                                          # 核对：mean|y| 应与 C++ 打印的一致
```

## 第二部分：Embedding 大表在 host 内存、乱序下标 gather

| 文件 | 内容 |
|---|---|
| `host_gather_pallas.py` | 表放在 host pinned 内存（`memory_kind="pinned_host"`），Pallas TensorCore kernel 用 `pl.HOST` 内存空间直接 DMA 按乱序下标取数据到 HBM，不经过 CPU。按对齐块取（见下） |

**用到的 API**

| 步骤 | API |
|---|---|
| 表放进 host pinned 内存 | `jax.device_put(table, jax.sharding.SingleDeviceSharding(dev, memory_kind="pinned_host"))` |
| 下标放进 SMEM（kernel 里用标量读） | `pl.pallas_call(..., grid_spec=pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=1, ...))`，下标作为第一个参数 |
| kernel 直接拿到 host 上的表（不整表拷到 HBM） | `in_specs=[pl.BlockSpec(memory_space=pl.HOST)]` |
| 输出留在 HBM、由 kernel 自己 DMA 写 | `out_specs=pl.BlockSpec(memory_space=pl.ANY)` |
| 按下标发 DMA：host → HBM | `pltpu.make_async_copy(src_ref.at[...], dst_ref.at[...], sem)`，`.start()` / `.wait()` |
| DMA 信号量（多个 DMA 同时在途） | `scratch_shapes=[pltpu.SemaphoreType.DMA((IN_FLIGHT,))]` |
| 对齐提示 | `pl.multiple_of(k * group, group)` |

```python
def kernel(idx_ref, table_ref, out_ref, sems):        # idx: SMEM，table: HOST，out: HBM
    def copy(i):
        src = table_ref.at[pl.ds(pl.multiple_of(idx_ref[i] * G, G), G)]   # 第 idx[i] 块
        dst = out_ref.at[pl.ds(pl.multiple_of(i * G, G), G)]              # 输出第 i 块
        return pltpu.make_async_copy(src, dst, sems.at[i % IN_FLIGHT])

    @pl.loop(0, n)
    def _(i):
        @pl.when(i >= IN_FLIGHT)
        def _():
            copy(i - IN_FLIGHT).wait()     # 复用信号量前等最早的那个完成
        copy(i).start()

    @pl.loop(n - IN_FLIGHT, n)
    def _(i):
        copy(i).wait()

out = pl.pallas_call(
    kernel,
    grid_spec=pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1, grid=(1,),
        in_specs=[pl.BlockSpec(memory_space=pl.HOST)],
        out_specs=pl.BlockSpec(memory_space=pl.ANY),
        scratch_shapes=[pltpu.SemaphoreType.DMA((IN_FLIGHT,))]),
    out_shape=jax.ShapeDtypeStruct((n * G, dim), table.dtype),
)(indices, table_host)
```

**验证状态**：CPU 上导出 TPU 版 StableHLO 成功：表参数标为 `mhlo.memory_kind="pinned_host"`，直接作为 `tpu_custom_call` 的输入，
中间没有整表拷贝（TPU 上编译后的 HLO 也确认没有拷贝）。

TPU 上实测（2026-11-17，v6e，JAX 0.11.2 / libtpu 0.0.50）：
- 运行时下标的 host DMA 以对齐块为单位：host 数组行宽为 4 KiB 的整数倍，一次取对齐的 8 行（一个 `(8,128)` tile 的高度），
  即每块 8 行 × 4 KiB = 32 KiB。
- 0.5 GiB 的表（f32 `[*, 1024]`）里乱序取 2048 块，与 numpy 按同样下标取的结果逐位一致。

---

## 第三部分：JIT 运行 / C++ 接口 / AOT 与 StableHLO

| 文件 | 内容 |
|---|---|
| `aot_stablehlo_export.py` | JIT、AOT（`lower().compile()`）、`jax.export` 导出 StableHLO 并序列化/反序列化执行；可在 CPU 机器上导出 TPU 产物 |
| `pjrt_run_stablehlo.cc` | C++ 通过 PJRT C API（`dlopen libtpu.so` → `GetPjrtApi`）编译并执行导出的 StableHLO |
| `third_party/xla/pjrt/c/pjrt_c_api.h` | PJRT C API 头文件（v0.116，来自本机 tpu-sync 编译用的 XLA） |
| `artifacts/`（运行后生成，不入库） | `model.stablehlo.mlir`（文本）、`.mlirbc`（bytecode）、`model.jax_exported`（序列化 Exported）、`compile_options.pb` |
| `check_cpp_vs_jax.py` | 用 libc `srand(0)/rand()` 逐位重建 C++ 的输入，在 JAX 里跑同一模型，核对 C++ 输出 |

运行命令见上面的"运行"。

**验证状态**：CPU 上导出 TPU 版 StableHLO 成功（模块含 Pallas 的 `stablehlo.custom_call @tpu_custom_call`），
序列化/反序列化成功；C++ 示例 `-Wall` 编译通过。

TPU 上实测（2026-11-17，v6e 单芯片，日志在 `runlogs/`）：
- Python：jit / AOT（`lower().compile()`）/ 反序列化 `Exported.call` 三者输出逐位相同（max|err| = 0）。
- C++：PJRT C API 0.116，编译并执行导出的 StableHLO，输出 `bf16[4096,1024]`，mean|y| = 0.2622；`check_cpp_vs_jax.py` 用逐位相同的输入在 JAX 里跑同一模型，mean|y| 也是 0.2622。
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
