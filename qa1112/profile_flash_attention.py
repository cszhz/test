"""01 的 profiler 版计时：用 jax.profiler 抓 trace，按 TPU 上的 device 时间比较各实现。

wall-clock（01b）包含 host 下发和同步开销；这里每个变体跑 N 次，读 xplane 里该 jit 模块在
/device:TPU:0 上每次执行的时长（中位数），并列出 kernel 内部的 op 拆分。
  python profile_flash_attention.py [--causal] [--iters 20] [--out /tmp/fa-prof]
汇总用 /zzlfs/ds41flash/tools/xplane_summary.py（本脚本自动调用）。
"""

import argparse
import glob
import math
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_kernel as splash,
)
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_mask as mask_lib,
)

ap = argparse.ArgumentParser()
ap.add_argument("--causal", action="store_true")
ap.add_argument("--iters", type=int, default=20)
ap.add_argument("--shape", default="64,2048,8,64", help="B,S,H,D")
ap.add_argument("--out", default="/tmp/fa-prof")
ap.add_argument("--block", type=int, default=0, help="大块 bq=bkv，默认 min(2048, S)")
ap.add_argument("--bkvc", type=int, default=512, help="大块的 block_kv_compute")
args = ap.parse_args()
B, S, H, D = map(int, args.shape.split(","))
causal = args.causal
flops = 4 * B * H * S * S * D * (0.5 if causal else 1.0)
scale = 1.0 / math.sqrt(D)

keys = jax.random.split(jax.random.key(0), 3)
q, k, v = (jax.random.normal(kk, (B, S, H, D), jnp.bfloat16) for kk in keys)
qh, kh, vh = (x.transpose(0, 2, 1, 3) for x in (q, k, v))
qs = (qh * scale).astype(qh.dtype)


def kernel(heads, bq, bkv, bkvc):
    one = mask_lib.CausalMask((S, S)) if causal else mask_lib.FullMask((S, S))
    return splash.make_splash_mha_single_device(
        mask_lib.MultiHeadMask([one] * heads),
        block_sizes=splash.BlockSizes(block_q=bq, block_kv=bkv, block_kv_compute=bkvc),
    )


def fa_xla_ref(q, k, v):
    s = jnp.einsum("bqhd,bkhd->bhqk", q, k, preferred_element_type=jnp.float32) * scale
    if causal:
        s = jnp.where(jnp.tril(jnp.ones((S, S), bool)), s, -1e30)
    return jnp.einsum("bhqk,bkhd->bqhd", jax.nn.softmax(s, axis=-1).astype(v.dtype), v)


def fa_xla_dpa(q, k, v):
    return jax.nn.dot_product_attention(q, k, v, is_causal=causal, implementation="xla")


BIG = args.block or min(2048, S)
SMALL = min(512, S)
BKVC = min(args.bkvc, BIG)
k512 = kernel(H, SMALL, SMALL, SMALL)
kbest = kernel(H, BIG, BIG, BKVC)
kfold = kernel(B * H, BIG, BIG, BKVC)


def fa_splash_vmap_small(q, k, v):
    return jax.vmap(k512)(q, k, v)


def fa_splash_vmap_big(q, k, v):
    return jax.vmap(kbest)(q, k, v)


def fa_splash_fold_big(q, k, v):
    r = lambda x: x.reshape(B * H, S, D)
    return kfold(r(q), r(k), r(v)).reshape(B, H, S, D)


def fa_splash_e2e_big(q, k, v):
    qh = (q * scale).astype(q.dtype).transpose(0, 2, 1, 3)
    out = jax.vmap(kbest)(qh, k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3))
    return out.transpose(0, 2, 1, 3)


variants = [
    (fa_xla_ref, (q, k, v)),
    (fa_xla_dpa, (q, k, v)),
    (fa_splash_vmap_small, (qs, kh, vh)),
    (fa_splash_vmap_big, (qs, kh, vh)),
    (fa_splash_fold_big, (qs, kh, vh)),
    (fa_splash_e2e_big, (q, k, v)),
]
jitted = [(f.__name__, jax.jit(f), xs) for f, xs in variants]
for _, fn, xs in jitted:  # 编译 + 预热，不进 trace
    for _ in range(3):
        jax.block_until_ready(fn(*xs))

os.makedirs(args.out, exist_ok=True)
with jax.profiler.trace(args.out):
    for _, fn, xs in jitted:
        for _ in range(args.iters):
            out = fn(*xs)
        jax.block_until_ready(out)

pb = sorted(glob.glob(f"{args.out}/plugins/profile/*/*.xplane.pb"), key=os.path.getmtime)[-1]
summary = subprocess.run(
    [sys.executable, "/zzlfs/ds41flash/tools/xplane_summary.py", pb, "/device:TPU:0"],
    capture_output=True, text=True,
).stdout
print(f"trace: {pb}\nshape [B,S,H,D]=[{B},{S},{H},{D}] causal={causal} small block={SMALL} big block={BIG} kv_compute={BKVC}\n")
print(f"{'variant':24s} {'device ms (median)':>18s} {'TFLOP/s':>8s}  n")
for name, _, _ in jitted:
    line = next((l for l in summary.splitlines() if f"jit_{name}" in l), None)
    if line is None:
        print(f"{name:24s} (not found in trace)")
        continue
    n = int(line.split("n=")[1].split()[0])
    med = float(line.split("med=")[1].split("ms")[0])
    print(f"{name:24s} {med:18.3f} {flops / med / 1e9:8.1f}  {n}")
print("\n--- xplane_summary (TPU:0) ---")
print(summary)
