"""splash forward 效率随 [B, S, H, D] 的变化（profiler device 时间，每组取最优块大小）。

固定每次计算量约 0.55 TFLOP（非 causal），改变 D、S、B×H 的分法，输出 kernel TFLOP/s。
  python shape_sweep.py [--causal] [--iters 10]
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
ap.add_argument("--causal", action="store_true")
ap.add_argument("--iters", type=int, default=10)
ap.add_argument("--out", default="/tmp/fa-shape-sweep")
args = ap.parse_args()
TARGET = 4 * 64 * 8 * 2048 * 2048 * 64  # 0.55 TFLOP


def configs():
    # (B, S, H, D)：B*H*S*S*D 保持不变
    out = []
    for D in (64, 128, 256):
        for S in (512, 1024, 2048, 4096, 8192):
            bh = TARGET // (4 * S * S * D)
            if bh < 1:
                continue
            H = 8 if bh >= 8 else bh  # kernel 只看 B*H 总数，固定 H=8
            out.append((bh // H, S, H, D))
    return out


def blocks(S):
    # 前面实测的经验：块越大越好，kv_compute 512；只试 3 种
    big = min(S, 2048)
    return sorted({(big, big, min(512, big)), (min(S, 1024), min(S, 1024), min(512, S)),
                   (min(S, 512), min(S, 1024), min(512, S))})


def make(B, S, H, D, bq, bkv, bkvc):
    one = (lambda: smk.CausalMask((S, S))) if args.causal else (lambda: smk.FullMask((S, S)))
    kern = sk.make_splash_mha(
        mask=smk.MultiHeadMask([one() for _ in range(H)]),
        block_sizes=sk.BlockSizes(block_q=bq, block_kv=bkv, block_kv_compute=bkvc),
        head_shards=1, q_seq_shards=1)
    return jax.jit(jax.vmap(kern))


runs = []
for (B, S, H, D) in configs():
    keys = jax.random.split(jax.random.key(0), 3)
    xs = [jax.random.normal(kk, (B, H, S, D), jnp.bfloat16) for kk in keys]
    for bq, bkv, bkvc in blocks(S):
        try:
            fn = make(B, S, H, D, bq, bkv, bkvc)
            for _ in range(2):
                jax.block_until_ready(fn(*xs))
        except Exception as e:
            print(f"skip {B,S,H,D} {bq,bkv,bkvc}: {type(e).__name__}")
            continue
        runs.append(((B, S, H, D), (bq, bkv, bkvc), fn, xs))

os.makedirs(args.out, exist_ok=True)
with jax.profiler.trace(args.out):
    for _, _, fn, xs in runs:
        for _ in range(args.iters):
            out = fn(*xs)
        jax.block_until_ready(out)

from jax._src.profiler import ProfileData
pb = sorted(glob.glob(f"{args.out}/plugins/profile/*/*.xplane.pb"), key=os.path.getmtime)[-1]
plane = next(p for p in ProfileData.from_file(pb).planes if p.name == "/device:TPU:0")
ops = sorted((e.start_ns, e.duration_ns) for l in plane.lines if l.name == "XLA Ops"
             for e in l.events if "splash_mha" in e.name)
assert len(ops) == len(runs) * args.iters, (len(ops), len(runs))
best = {}
for i, (shape, blk, _, _) in enumerate(runs):
    d = sorted(x[1] for x in ops[i * args.iters:(i + 1) * args.iters])[args.iters // 2] / 1e6
    if shape not in best or d < best[shape][0]:
        best[shape] = (d, blk)

print(f"causal={args.causal}，每组计算量 {TARGET / 1e12:.2f} TFLOP（非 causal 口径），kernel device 时间，最优块大小")
print(f"{'B':>5} {'S':>5} {'H':>3} {'D':>4} {'B*H':>5}  {'kernel ms':>9} {'TFLOP/s':>8}  best q/kv/kvc")
fl = TARGET * (0.5 if args.causal else 1.0)
for shape in sorted(best, key=lambda s: (s[3], s[1], s[2])):
    d, blk = best[shape]
    B, S, H, D = shape
    print(f"{B:5d} {S:5d} {H:3d} {D:4d} {B * H:5d}  {d:9.3f} {fl / d / 1e9:8.1f}  {blk[0]}/{blk[1]}/{blk[2]}")
