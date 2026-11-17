"""Splash attention 整颗芯片（2 个 TensorCore）前向 / 前向+反向计时，可单独运行。

配置：[B, H, S, D] = [128, 2, 1024, 256] bf16，causal，前向块 1024/1024/512，反向块 1024，fused bwd。
batch 用 shard_map 切到同一芯片的两个核（各 64），时间取 XProf 里两核中较慢者（20 次中位数）。
参考结果（TPU7x）：fwd ≈ 0.384 ms，fwd+bwd ≈ 0.967 ms。

  source /zzlfs/temu/tools/sc_repro/env.sh
  /zzlfs/temu/.venv-epic/bin/python splash_fwd_bwd_chip.py
"""

import glob
import math
import os

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as sk
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as smk
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

B, H, S, D = 128, 2, 1024, 256
scale = 1.0 / math.sqrt(D)

# ---- splash kernel：causal，fused bwd ----
block_sizes = sk.BlockSizes(
    block_q=1024, block_kv=1024, block_kv_compute=512,
    block_q_dkv=1024, block_kv_dkv=1024, block_kv_dkv_compute=1024,
    use_fused_bwd_kernel=True,
)
mask = smk.MultiHeadMask([smk.CausalMask((S, S)) for _ in range(H)])
kernel = sk.make_splash_mha(mask=mask, block_sizes=block_sizes,
                            head_shards=1, q_seq_shards=1)


def attn(q, k, v):                     # q/k/v: [B, H, S, D]
    # splash 不做缩放，先把 q 乘 1/sqrt(D)
    return jax.vmap(kernel)((q * scale).astype(q.dtype), k, v)


def attn_fwd_bwd(q, k, v, do):         # 训练：前向 + 反向（dq, dk, dv）
    out, vjp = jax.vjp(attn, q, k, v)
    return (out, *vjp(do))


# ---- 同一颗芯片的两个核（coords 相同，core_on_chip = 0 / 1），batch 维切两份 ----
d0 = jax.devices()[0]
cores = [d for d in jax.devices() if d.coords == d0.coords]
mesh = Mesh(np.array(cores), ("core",))
spec = P("core")
fwd = jax.jit(jax.shard_map(attn, mesh=mesh, in_specs=(spec,) * 3,
                            out_specs=spec, check_vma=False))
fwd_bwd = jax.jit(jax.shard_map(attn_fwd_bwd, mesh=mesh, in_specs=(spec,) * 4,
                                out_specs=(spec,) * 4, check_vma=False))

ks = jax.random.split(jax.random.key(0), 4)
q, k, v, do = (jax.device_put(jax.random.normal(kk, (B, H, S, D), jnp.bfloat16),
                              NamedSharding(mesh, spec)) for kk in ks)

# 预热（编译不计入）
for _ in range(3):
    jax.block_until_ready(fwd(q, k, v))
    jax.block_until_ready(fwd_bwd(q, k, v, do))

# ---- XProf：抓 device 时间 ----
ITERS, OUT = 20, "/tmp/splash-chip-prof"
with jax.profiler.trace(OUT):
    for _ in range(ITERS):
        o = fwd(q, k, v)
    jax.block_until_ready(o)
    for _ in range(ITERS):
        o = fwd_bwd(q, k, v, do)
    jax.block_until_ready(o)

from jax._src.profiler import ProfileData

pb = sorted(glob.glob(f"{OUT}/plugins/profile/*/*.xplane.pb"), key=os.path.getmtime)[-1]
planes = {p.name: p for p in ProfileData.from_file(pb).planes}


def module_times(dev):                 # 该核上每次 jit 模块执行的 (开始, 时长)
    p = planes[f"/device:TPU:{dev.id}"]
    return sorted((e.start_ns, e.duration_ns) for l in p.lines
                  if l.name == "XLA Modules" for e in l.events)


m0, m1 = module_times(cores[0]), module_times(cores[1])
assert len(m0) == len(m1) == 2 * ITERS, (len(m0), len(m1))
slowest = [max(a[1], b[1]) / 1e6 for a, b in zip(m0, m1)]   # 两核取慢的
med = lambda xs: sorted(xs)[len(xs) // 2]
print(f"[B,H,S,D]=[{B},{H},{S},{D}] causal, 2 cores {[d.id for d in cores]}")
print(f"fwd      {med(slowest[:ITERS]):.3f} ms")
print(f"fwd+bwd  {med(slowest[ITERS:]):.3f} ms")
