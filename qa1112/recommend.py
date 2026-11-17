"""固定 token 数（B*S）和 hidden（H*D），比较 head 分法与块大小的实际耗时（前向、前向+反向 fused）。
  python recommend.py [--causal]
"""
import argparse, glob, math, os
import jax, jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as sk
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as smk

ap = argparse.ArgumentParser()
ap.add_argument("--causal", action="store_true")
ap.add_argument("--iters", type=int, default=20)
ap.add_argument("--out", default="/tmp/fa-recommend")
args = ap.parse_args()

CFGS = [  # name, B, S, H, D, (bq, bkv, bkvc)
    ("A 基准 H4xD128 块512/1024/512", 128, 1024, 4, 128, (512, 1024, 512)),
    ("B 调块 H4xD128 块1024/1024/512", 128, 1024, 4, 128, (1024, 1024, 512)),
    ("C 少头 H2xD256 块1024/1024/512", 128, 1024, 2, 256, (1024, 1024, 512)),
    ("D 多头 H8xD64  块1024/1024/512", 128, 1024, 8, 64, (1024, 1024, 512)),
]


def build(B, S, H, D, blk):
    bq, bkv, bkvc = blk
    one = (lambda: smk.CausalMask((S, S))) if args.causal else (lambda: smk.FullMask((S, S)))
    kern = sk.make_splash_mha(
        mask=smk.MultiHeadMask([one() for _ in range(H)]),
        block_sizes=sk.BlockSizes(block_q=bq, block_kv=bkv, block_kv_compute=bkvc,
                                  block_q_dkv=min(S, 1024), block_kv_dkv=min(S, 1024),
                                  block_kv_dkv_compute=min(S, 1024), use_fused_bwd_kernel=True),
        head_shards=1, q_seq_shards=1)
    scale = 1.0 / math.sqrt(D)
    f = lambda q, k, v: jax.vmap(kern)((q * scale).astype(q.dtype), k, v)

    def fb(q, k, v, do):
        out, vjp = jax.vjp(f, q, k, v)
        return (out, *vjp(do))
    return jax.jit(f), jax.jit(fb)


runs = []
for name, B, S, H, D, blk in CFGS:
    ks = jax.random.split(jax.random.key(0), 4)
    q, k, v, do = (jax.random.normal(kk, (B, H, S, D), jnp.bfloat16) for kk in ks)
    f, fb = build(B, S, H, D, blk)
    for _ in range(3):
        jax.block_until_ready(f(q, k, v)); jax.block_until_ready(fb(q, k, v, do))
    runs += [(name, "fwd", f, (q, k, v)), (name, "fwd+bwd", fb, (q, k, v, do))]

os.makedirs(args.out, exist_ok=True)
with jax.profiler.trace(args.out):
    for _, _, fn, xs in runs:
        for _ in range(args.iters):
            o = fn(*xs)
        jax.block_until_ready(o)

from jax._src.profiler import ProfileData
pb = sorted(glob.glob(f"{args.out}/plugins/profile/*/*.xplane.pb"), key=os.path.getmtime)[-1]
plane = next(p for p in ProfileData.from_file(pb).planes if p.name == "/device:TPU:0")
mods = sorted((e.start_ns, e.duration_ns) for l in plane.lines if l.name == "XLA Modules" for e in l.events)
assert len(mods) == len(runs) * args.iters
res = {}
for i, (name, kind, _, _) in enumerate(runs):
    d = sorted(x[1] for x in mods[i * args.iters:(i + 1) * args.iters])[args.iters // 2] / 1e6
    res.setdefault(name, {})[kind] = d
print(f"causal={args.causal}  token 数 B*S=131072，hidden H*D=512，device 时间（含 q 乘 scale）")
base = res[CFGS[0][0]]
print(f"{'配置':34s} {'前向 ms':>8s} {'vs A':>6s} {'前向+反向 ms':>12s} {'vs A':>6s}")
for name, r in res.items():
    print(f"{name:34s} {r['fwd']:8.3f} {base['fwd'] / r['fwd']:5.2f}x {r['fwd+bwd']:12.3f} {base['fwd+bwd'] / r['fwd+bwd']:5.2f}x")
