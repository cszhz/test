"""整颗芯片（2 个 TensorCore）跑 splash attention：batch 按 2 份切到同一芯片的两个核上（shard_map）。

对比单核（整个 batch 在 1 个核上）。时间取 profiler 里两个核上该 jit 模块执行时长的较大者（中位数）。
  python profile_chip.py [--shape 128,1024,2,256] [--no-causal]
"""

import argparse
import glob
import math
import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as sk
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as smk
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

ap = argparse.ArgumentParser()
ap.add_argument("--shape", default="128,1024,2,256", help="B,S,H,D")
ap.add_argument("--bq", type=int, default=1024)
ap.add_argument("--bkv", type=int, default=1024)
ap.add_argument("--bkvc", type=int, default=512)
ap.add_argument("--bs", type=int, default=1024, help="反向块大小")
ap.add_argument("--no-causal", action="store_true")
ap.add_argument("--iters", type=int, default=20)
ap.add_argument("--out", default="/tmp/fa-prof-chip")
args = ap.parse_args()
B, S, H, D = map(int, args.shape.split(","))
causal = not args.no_causal
scale = 1.0 / math.sqrt(D)

# 同一颗芯片上的两个核：coords 相同、core_on_chip 为 0 / 1
d0 = jax.devices()[0]
chip = [d for d in jax.devices() if d.coords == d0.coords]
assert len(chip) == 2, chip
mesh = Mesh(np.array(chip), ("core",))

one = (lambda: smk.CausalMask((S, S))) if causal else (lambda: smk.FullMask((S, S)))
kernel = sk.make_splash_mha(
    mask=smk.MultiHeadMask([one() for _ in range(H)]),
    block_sizes=sk.BlockSizes(
        block_q=args.bq, block_kv=args.bkv, block_kv_compute=args.bkvc,
        block_q_dkv=args.bs, block_kv_dkv=args.bs, block_kv_dkv_compute=args.bs,
        use_fused_bwd_kernel=True),
    head_shards=1, q_seq_shards=1)


def attn(q, k, v):
    return jax.vmap(kernel)((q * scale).astype(q.dtype), k, v)


def fwd_bwd(f):
    def run(q, k, v, do):
        out, vjp = jax.vjp(f, q, k, v)
        return (out, *vjp(do))
    return run


spec = P("core")
attn_chip = jax.shard_map(attn, mesh=mesh, in_specs=(spec,) * 3, out_specs=spec, check_vma=False)
fb_chip = jax.shard_map(fwd_bwd(attn), mesh=mesh, in_specs=(spec,) * 4,
                        out_specs=(spec,) * 4, check_vma=False)

ks = jax.random.split(jax.random.key(0), 4)
host = [jax.random.normal(kk, (B, H, S, D), jnp.bfloat16) for kk in ks]
one_core = [jax.device_put(x, d0) for x in host]
sharded = [jax.device_put(x, NamedSharding(mesh, spec)) for x in host]

runs = [
    ("单核 fwd", jax.jit(attn), one_core[:3]),
    ("单核 fwd+bwd", jax.jit(fwd_bwd(attn)), one_core),
    ("整芯片(2核) fwd", jax.jit(attn_chip), sharded[:3]),
    ("整芯片(2核) fwd+bwd", jax.jit(fb_chip), sharded),
]
for _, fn, xs in runs:
    for _ in range(3):
        jax.block_until_ready(fn(*xs))
# 数值：整芯片结果 = 单核结果
for a, b in zip(runs[3][1](*sharded), runs[1][1](*one_core)):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    assert float(np.max(np.abs(a - b))) < 1e-6, float(np.max(np.abs(a - b)))

os.makedirs(args.out, exist_ok=True)
with jax.profiler.trace(args.out):
    for _, fn, xs in runs:
        for _ in range(args.iters):
            o = fn(*xs)
        jax.block_until_ready(o)

from jax._src.profiler import ProfileData
pb = sorted(glob.glob(f"{args.out}/plugins/profile/*/*.xplane.pb"), key=os.path.getmtime)[-1]
planes = {p.name: p for p in ProfileData.from_file(pb).planes}


def modules(core):
    p = planes[f"/device:TPU:{chip[core].id}"]
    return sorted((e.start_ns, e.duration_ns) for l in p.lines if l.name == "XLA Modules" for e in l.events)


m0, m1 = modules(0), modules(1)
n = args.iters
med = lambda xs: sorted(xs)[len(xs) // 2] / 1e6
# 核 0 上：单核 fwd、单核 fwd+bwd、整芯片 fwd、整芯片 fwd+bwd；核 1 上只有后两组
assert len(m0) == 4 * n and len(m1) == 2 * n, (len(m0), len(m1))
res = [
    ("单核 fwd", med([d for _, d in m0[0:n]])),
    ("单核 fwd+bwd", med([d for _, d in m0[n:2 * n]])),
    ("整芯片(2核) fwd", med([max(a[1], b[1]) for a, b in zip(m0[2 * n:3 * n], m1[0:n])])),
    ("整芯片(2核) fwd+bwd", med([max(a[1], b[1]) for a, b in zip(m0[3 * n:4 * n], m1[n:2 * n])])),
]
print(f"shape [B,H,S,D]=[{B},{H},{S},{D}] causal={causal} 块 {args.bq}/{args.bkv}/{args.bkvc} bwd {args.bs}，"
      f"芯片 coords={d0.coords}，device {[d.id for d in chip]}")
for name, ms in res:
    print(f"  {name:22s} {ms:7.3f} ms")
print(f"  加速比 fwd {res[0][1] / res[2][1]:.2f}x，fwd+bwd {res[1][1] / res[3][1]:.2f}x")
