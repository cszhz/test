"""splash attention 训练场景（前向 + 反向）的 profiler 计时：fused bwd vs 非 fused，对照 XLA。

配置与用户写法一致：make_splash_mha(head_shards=1, q_seq_shards=1)，所有块 = bs，vmap over batch。
  python profile_fwd_bwd.py [--shape 128,1024,4,128] [--bs 1024] [--no-causal]
"""

import argparse
import glob
import math
import os

import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as sk
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as smk

ap = argparse.ArgumentParser()
ap.add_argument("--shape", default="128,1024,4,128", help="B,S,H,D")
ap.add_argument("--bs", type=int, default=1024)
ap.add_argument("--bq", type=int, default=0, help="前向 block_q，默认 = bs")
ap.add_argument("--bkv", type=int, default=0, help="前向 block_kv，默认 = bs")
ap.add_argument("--bkvc", type=int, default=0, help="前向 block_kv_compute，默认 = bs")
ap.add_argument("--no-causal", action="store_true")
ap.add_argument("--iters", type=int, default=20)
ap.add_argument("--out", default="/tmp/fa-prof-bwd")
args = ap.parse_args()
B, S, H, D = map(int, args.shape.split(","))
causal = not args.no_causal
bs = args.bs
bq, bkv, bkvc = args.bq or bs, args.bkv or bs, args.bkvc or bs
scale = 1.0 / math.sqrt(D)
fwd_flops = 4 * B * H * S * S * D * (0.5 if causal else 1.0)

keys = jax.random.split(jax.random.key(0), 4)
q, k, v, do = (jax.random.normal(kk, (B, H, S, D), jnp.bfloat16) for kk in keys)


def splash(fused: bool):
    block_sizes = sk.BlockSizes(
        block_q=bq, block_kv=bkv, block_kv_compute=bkvc,
        block_q_dkv=bs, block_kv_dkv=bs, block_kv_dkv_compute=bs,
        **({"use_fused_bwd_kernel": True} if fused else
           {"block_q_dq": bs, "block_kv_dq": bs}),
    )
    one = (lambda: smk.CausalMask((S, S))) if causal else (lambda: smk.FullMask((S, S)))
    mask = smk.MultiHeadMask([one() for _ in range(H)])
    kernel = sk.make_splash_mha(mask=mask, block_sizes=block_sizes,
                                head_shards=1, q_seq_shards=1)
    return lambda q, k, v: jax.vmap(kernel)((q * scale).astype(q.dtype), k, v)


def xla(q, k, v):
    s = jnp.einsum("bhqd,bhkd->bhqk", q, k, preferred_element_type=jnp.float32) * scale
    if causal:
        s = jnp.where(jnp.tril(jnp.ones((S, S), bool)), s, -1e30)
    return jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(s, axis=-1).astype(v.dtype), v)


def fwd_bwd(f):
    def run(q, k, v, do):
        out, vjp = jax.vjp(f, q, k, v)
        return (out, *vjp(do))
    return run


impls = {"splash_fused": splash(True), "splash_unfused": splash(False), "xla": xla}
fns = {}
for name, f in impls.items():
    fns[f"fwd_{name}"] = (jax.jit(f, inline=False), (q, k, v))
    fns[f"fwdbwd_{name}"] = (jax.jit(fwd_bwd(f)), (q, k, v, do))
# 名字带进 jit 模块名，方便在 trace 里区分
fns = {n: (jax.jit(jax.named_call(fn, name=n)), xs) for n, (fn, xs) in fns.items()}
for n, (fn, xs) in fns.items():
    for _ in range(3):
        jax.block_until_ready(fn(*xs))

# 数值：splash 梯度 vs XLA 梯度
ref = fns["fwdbwd_xla"][0](q, k, v, do)
for n in ("fwdbwd_splash_fused", "fwdbwd_splash_unfused"):
    got = fns[n][0](q, k, v, do)
    errs = [float(jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32)))) for a, b in zip(got, ref)]
    print(f"{n:24s} max|err| out/dq/dk/dv = " + " / ".join(f"{e:.3g}" for e in errs))

os.makedirs(args.out, exist_ok=True)
order = list(fns)
with jax.profiler.trace(args.out):
    for n in order:
        fn, xs = fns[n]
        with jax.profiler.TraceAnnotation(n):
            for _ in range(args.iters):
                out = fn(*xs)
            jax.block_until_ready(out)

from jax._src.profiler import ProfileData
pb = sorted(glob.glob(f"{args.out}/plugins/profile/*/*.xplane.pb"), key=os.path.getmtime)[-1]
plane = next(p for p in ProfileData.from_file(pb).planes if p.name == "/device:TPU:0")
lines = {l.name: l for l in plane.lines}
mods = sorted((e.start_ns, e.duration_ns, e.name) for e in lines["XLA Modules"].events)
ops = sorted((e.start_ns, e.duration_ns, e.name.split(".")[0].split(" ")[0]) for e in lines["XLA Ops"].events)
# 模块按执行顺序分组：每个变体 iters 次
print(f"\nshape [B,H,S,D]=[{B},{H},{S},{D}] causal={causal} fwd q/kv/kvc={bq}/{bkv}/{bkvc} bwd={bs}  trace {pb}")
print(f"{'variant':24s} {'device ms':>10s} {'TFLOP/s':>8s}   op 拆分（每次）")
assert len(mods) == len(order) * args.iters, (len(mods), len(order))
for i, n in enumerate(order):
    group = mods[i * args.iters:(i + 1) * args.iters]
    ms = sorted(d for _, d, _ in group)[len(group) // 2] / 1e6
    fl = fwd_flops * (3.5 if n.startswith("fwdbwd") else 1.0)  # 反向约 2.5 倍前向
    t0, t1 = group[0][0], group[-1][0] + group[-1][1]
    agg = {}
    for s, d, name in ops:
        if t0 <= s < t1:
            agg[name] = agg.get(name, 0) + d
    top = sorted(agg.items(), key=lambda kv: -kv[1])[:4]
    br = ", ".join(f"{k} {v / 1e6 / args.iters:.3f}" for k, v in top)
    print(f"{n:24s} {ms:10.3f} {fl / ms / 1e9:8.1f}   {br}")
