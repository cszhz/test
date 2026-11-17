# 算子在 TPU 上的实现（问答 2026-11-12，第一部分：算子）

Attention shape：`[batch, seq, head, dim] = [64, 2048, 8, 64]`，bf16。环境：JAX 0.11.2 / libtpu 0.0.50（venv `/disk/venvs/qa1112-jax`，`env.sh` 里的 `$PY`）。

| 文件 | 算子 | TPU 实现 |
|---|---|---|
| `flash_attention_mha.py` | Flash Attention（MHA） | Pallas `splash_attention`（`make_splash_mha_single_device` + `vmap` 处理 batch） |
| `paged_attention.py` | PageAttention | Pallas `ragged_paged_attention`（prefill/decode 混合、变长，vLLM TPU 后端在用） |
| `rmsnorm.py` | RmsNorm | 推荐纯 jnp（XLA 自动融合）；附 Pallas 版 fused residual-add + RmsNorm |
| `moe_grouped_gemm.py` | MoE（对应 SonicMoE） | top-k 路由 + 按专家排序 + 分组 GEMM：Pallas `megablox.gmm` 或 XLA `jax.lax.ragged_dot` |
| `common.py` | 公共 | 计时、误差比较、参数 |
| `bench_flash_attention.py` / `profile_flash_attention.py` / `profile_fwd_bwd.py` | Flash Attention 实测 | 块大小扫描 / profiler 拆 op / 前向+反向 |
| `shape_sweep.py` / `recommend.py` / `profile_chip.py` | Flash Attention 实测 | 形状扫描 / 建议值对比 / 整芯片（2 个 TensorCore） |
| `splash_chip_sweep.py` / `splash_chip_sweep_v2.py` / `splash_fwd_bwd_chip.py` | Flash Attention 实测 | 整芯片下的块大小 / 形状扫描、前向+反向 |
| `xplane_ops_by_module.py` | 工具 | 从 profiler 的 xplane 按模块拆 op 时间 |

## 运行

```bash
cd ~/temu
source env.sh        # 设 $PY；并把进程限定在 2 号芯片（本机其余芯片有服务在跑），独占机器时可不 source、自己设 PY
# TPU 上（真实 shape，先与参考实现比数值再计时；需要一台没在跑服务的 TPU 机器）
$PY flash_attention_mha.py [--causal] [--block-q 2048 --block-kv 2048]
$PY bench_flash_attention.py [--causal]     # 块大小扫描（wall-clock）
$PY profile_flash_attention.py [--causal]   # profiler：device 时间 + op 拆分
$PY paged_attention.py
$PY rmsnorm.py
$PY moe_grouped_gemm.py [--impl gmm|ragged] [--tokens 8192 --hidden 2048 --inter 1024 --experts 64 --top-k 8] [--tile 512 512 512]
# CPU 上检查正确性（小 shape，Pallas interpret 模式）
JAX_PLATFORMS=cpu $PY flash_attention_mha.py --small [--causal]
```

## 验证状态

- CPU（`--small`，interpret 模式）：splash MHA（causal / 非 causal）、fused residual+RmsNorm、MoE（gmm、ragged_dot）都与参考实现一致。
- `ragged_paged_attention` 依赖 TPU 的 scalar prefetch / DMA，CPU 上只验证了参考实现和页表/布局构造。
- TPU 实测：Flash Attention 已测（见下方"Flash Attention 实测"），其余算子待测。

### Flash Attention 实测（2026-11-13，TPU7x 单个 TensorCore，`[64,2048,8,64]` bf16）

用 profiler 看 device 时间（`profile_flash_attention.py`，按模块拆 op 用 `xplane_ops_by_module.py`）；块大小扫描用 `bench_flash_attention.py`
（wall-clock，和 device 时间相差 <0.01 ms）。

| 实现 | 非 causal：整体 / 其中 kernel | causal：整体 / 其中 kernel |
|---|---|---|
| XLA einsum+softmax（= `jax.nn.dot_product_attention(xla)`，HLO 相同） | 5.73 ms（96 TFLOP/s） | 5.82 ms |
| splash vmap，块 512（原脚本默认） | 7.83 / 6.87 ms | 6.74 / 5.78 ms |
| splash vmap，块 2048 / kv_compute 512 | 4.09 / 3.12 ms | 4.46 / 3.50 ms |
| **splash，B·H 折进 head 维，块 2048** | **3.52 / 3.10 ms（kernel 177 TFLOP/s）** | **3.90 / 3.48 ms** |
| splash 端到端（含 scale 与 [B,S,H,D] 转置），块 2048 | 3.69 / 3.12 ms | 4.06 / 3.50 ms |

- 块大小影响一倍多：512→2048，kernel 6.87→3.12 ms；`block_kv_compute` 取 256/512 最好。
- vmap 版多出 4 次 `copy`（0.96 ms），是 kernel 要求的布局和输入不同；折叠 / 端到端版里这一步变成 SparseCore 上的
  `sparse-core-data-format-call`（0.2–0.4 ms）。
- causal 在 S=2048 下省不了：块小到能跳过上三角时，每块的固定开销比省下的多；块 2048 时只能逐元素 mask，比非 causal 还慢约 12%。
- kernel 177 TFLOP/s 约是单核 bf16 峰值的 15%：head_dim=64 时 QKᵀ 的收缩维和 PV 的输出维只用到 MXU 的 1/4（按 256×256 MXU 估算），
  另外每个分数都要做一次 exp。

**`[128,1024,4,128]` bf16**（`--shape 128,1024,4,128`，同样 0.275 TFLOP/次非 causal，是上面的一半）

| 实现 | 非 causal：整体 / 其中 kernel | causal：整体 / 其中 kernel |
|---|---|---|
| XLA einsum+softmax（= `dot_product_attention(xla)`） | 1.81 ms（152 TFLOP/s） | 1.88 ms |
| splash vmap，块 512 | 1.81 / 1.81 ms | 1.75 / 1.74 ms |
| **splash vmap / 折叠，块 1024（= S）** | **0.95 / 0.95 ms（289 TFLOP/s）** | **1.05–1.06 / 1.05 ms** |
| splash 端到端（含 scale 与转置），块 1024 | 1.32 / 0.95 ms | 1.43 / 1.06 ms |

- head_dim=128 时 vmap 版没有布局 copy（kernel 布局和输入一致），vmap 与折叠一样快。非 causal 用 `block_kv_compute=1024`，causal 用 512。
- 端到端版多出 0.37 ms：scale 0.08 ms + SparseCore 布局转换 0.09 ms，其余是模块内 TensorCore 等 SparseCore 的空隙。
  实际模型里转置通常能融合进前面的 QKV 投影，按 kernel 时间看更接近。
- 每 FLOP 效率是 head_dim=64 时的 1.6 倍（289 vs 177 TFLOP/s）：收缩维 / 输出维从 64 变成 128。约为单核峰值的 25%。
- causal 同样省不了：S=1024 时块再小（512）就更慢，块 = S 时只能逐元素 mask，比非 causal 慢约 11%。

**训练（前向 + 反向），`[128,1024,4,128]` causal，所有块 1024**（`profile_fwd_bwd.py`；`make_splash_mha(head_shards=1, q_seq_shards=1)`
与 `make_splash_mha_single_device` 等价，dkv/dq 块大小和 `use_fused_bwd_kernel` 只影响反向）

| 实现 | 前向 | 前向 + 反向 | op 拆分（前向 + 反向） |
|---|---|---|---|
| splash，fused bwd | 1.15 ms | **3.17 ms** | fwd（存 logsumexp）1.12 + dq/dk/dv 一个 kernel 1.62 + copy 等 0.25 |
| splash，非 fused bwd | 1.15 ms | 4.01 ms | fwd 1.12 + dkv 1.28 + dq 1.09 + copy 等 0.30 |
| XLA einsum+softmax | 1.57 ms | 5.37 ms | 融合 fusion 3.25 + 卷积 2.12 |

- fused bwd 把 dq 并进 dkv kernel，一遍扫完 K/V，比分开的两个 kernel 少 0.74 ms（前向 + 反向快 21%），是 XLA 的 1.7 倍快。
- 前向含 scale（`q * 1/sqrt(dim)`，0.08 ms）；splash 梯度与 XLA 梯度的最大误差 dq 0.039 / dk、dv 0.031（bf16）。

**形状对效率的影响**（`shape_sweep.py`，非 causal，每组计算量都是 0.55 TFLOP，kernel device 时间，每组取最优块大小，单位 TFLOP/s）

| D \ S | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|
| 64 | 71 | 140 | 176 | 182 | 186 |
| 128 | 143 | 279 | 354 | 362 | 370 |
| 256 | 221 | 451 | 472 | 648 | 667 |

- head_dim 影响最大：64→128 约翻倍；128→256 在 S≥4096 时约 1.8 倍、S=2048 时只有 1.33 倍（MXU 收缩维 / 输出维用得更满）。总 hidden（H×D）固定时，少 head、大 D 更快。
- S 越长效率越高：S=512 时只有 S≥2048 的 40%（每个 Q 块只循环一两次 KV，固定开销占比大）。但计算量随 S² 增长，
  **不要为了效率把短样本拼长或补齐**（两条 1024 拼成 2048 计算量翻倍，效率只高 1.27 倍，总时间更长）。S 按数据定。
- B×H 只要总数够就行（低到 8 也不掉）；B 和 H 怎么分对 kernel 没区别。
- 块大小：S≤2048 用 q = kv = S、kv_compute 512；D=256、S=2048 时 2048 块编译失败（VMEM 不够），用 1024。

**建议值实测**（`recommend.py`：token 数 B×S = 131072、hidden H×D = 512 固定，S=1024，device 时间，反向 fused）

| 配置 | 非 causal：前向 / 前向+反向 | causal：前向 / 前向+反向 |
|---|---|---|
| A 基准：H4×D128，块 512/1024/512 | 1.32 / 3.22 ms | 1.32 / 3.34 ms |
| B 只调块：H4×D128，块 1024/1024/512 | 1.07 / 2.97 ms（1.24× / 1.09×） | 1.15 / 3.17 ms（1.15× / 1.05×） |
| C 少头大 D：H2×D256，块 1024/1024/512 | **0.70 / 1.78 ms（1.90× / 1.81×）** | **0.75 / 1.91 ms（1.75× / 1.74×）** |
| D 多头小 D：H8×D64，块 1024/1024/512 | 2.91 / 7.65 ms（0.45× / 0.42×） | 3.05 / 8.00 ms（0.43× / 0.42×） |

- 块大小不改模型，可以直接用（前向快 15–24%）；改 head 分法会改变模型结构（同样 hidden 下头数减半），要看精度能否接受。

**整颗芯片（2 个 TensorCore）**（`profile_chip.py`：batch 用 `shard_map` 切到同一芯片的两个核，各跑 B/2；时间取两核中较慢的那个）

`[128,2,1024,256]` causal，块 1024/1024/512，fused bwd：

| | 前向 | 前向 + 反向 |
|---|---|---|
| 单核（整个 batch） | 0.754 ms | 1.916 ms |
| **整芯片（2 核各 64）** | **0.384 ms** | **0.967 ms** |

- 加速 1.97× / 1.98×，两核之间不需要通信（attention 按 batch 独立），结果与单核逐位一致。和 GPU 按"卡对芯片"比时用这一行。
- 以上都含 q 乘 scale（单核约 0.083 ms 前向）；GPU 的 FA 通常在 kernel 内做缩放，对比时可扣掉。

## 要点

**Flash Attention**
- kernel 输入 `[heads, seq, dim]`（head_dim 在最内层），外部 `[B, S, H, D]` 先转成 `[B, H, S, D]`，batch 用 `vmap`。
- kernel 不乘 `1/sqrt(dim)`，调用前乘在 q 上。
- mask 用 `MultiHeadMask`（每个 head 一个 `CausalMask` / `FullMask` / `LocalMask`），按块稀疏跳过不需要算的块。
- `head_dim=64`：QKᵀ 的收缩维和 PV 的输出维只用到 MXU 的 1/4（256×256 MXU），是硬件限制；这个 shape 下 attention 计算量
  不大（非 causal 约 0.55 TFLOP/次），一般不是瓶颈。这个 shape 下 `block_q`/`block_kv` 用 2048、`block_kv_compute` 512 最快（见实测）。

**PageAttention**
- 布局和 CUDA 版不同：`kv_pages = [total_pages, page_size, 2*num_kv_heads, head_dim]`，K 在偶数位、V 在奇数位；
  所有请求的 query 首尾拼接，用 `cu_q_lens` 前缀和定位，`kv_lens` + `page_indices` 描述每个请求的 KV。
- causal mask 由 kernel 按 `kv_len - q_len` 偏移自动生成；支持 GQA（q 头数为 kv 头数的整数倍）、sliding window、soft cap、FP8 KV（k/v scale）。
- `ragged_paged_attention` 要求 head_dim 是 128 的倍数（head_dim=64 时 TPU 上编译报 `shape[1] % arr.shape[1] == 0`）。
  `paged_attention.py` 把 q 和 kv_pages 的最后一维补零到 128、`sm_scale` 仍按真实 dim 取 `dim**-0.5`，输出再切回 64，结果与参考一致；
  KV cache 因此按 128 占 HBM（head_dim=64 时多一倍），规划容量时按补齐后的大小算。

**RmsNorm**
- 用 jnp 写就够了，XLA 会和前后的 elementwise、残差融合；手写 Pallas 只在需要和量化/残差等强行融合时才有意义。
- Pallas 写法：按行分块（`block_rows × hidden`），hidden 在 lane 上（128 的倍数），fp32 里做平方和与 rsqrt。

**MoE（SonicMoE）**
- 分组 GEMM：`megablox.gmm(lhs[m,k], rhs[g,k,n], group_sizes[g], tiling=(tm,tk,tn))`，tile 取 128 的倍数，可调；
  或 `jax.lax.ragged_dot`（XLA 原生，不用调参）。
- token 按 tile 取整的思路对应到 gmm 的 tm 对齐。
- 专家并行（多卡）：dispatch/combine 用 `jax.lax.ragged_all_to_all`（或 shard_map + all_to_all）；本目录只做单卡。
  计算与通信重叠可参考 vllm-torchtpu 的 fused EP MoE（`MOE_FUSED_EP_KERNEL_IMPL=push`，W4A8）。

**PyTorch 代码**
- 不必整体改写成 JAX：用 torch_tpu 跑 PyTorch，需要的 kernel 用 `torch.tpu.pallas.jax_op` 把 JAX Pallas kernel 包成 PyTorch 算子
  （vllm-torchtpu 里的例子：`src/vllm_torchtpu/layers/adapter/router_topk.py`）。

---

# 第二部分：Embedding 大表在 host 内存、乱序下标 gather

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

# 第三部分：JIT 运行 / C++ 接口 / AOT 与 StableHLO —— 都能接

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
