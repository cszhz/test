"""01 的性能扫描：splash attention 各种调用方式 / 块大小 vs XLA。

shape 默认 [B, S, H, D] = [64, 2048, 8, 64] bf16。每个变体先和 fp32 参考比对，再计时。
  python bench_flash_attention.py [--causal] [--iters 50] [--quick]
TPU 环境变量见 /zzlfs/temu/tools/sc_repro/env.sh。
"""

import argparse
import itertools
import math

import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_kernel as splash,
)
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_mask as mask_lib,
)

from common import bench, max_abs_err

ap = argparse.ArgumentParser()
ap.add_argument("--causal", action="store_true")
ap.add_argument("--iters", type=int, default=50)
ap.add_argument("--quick", action="store_true", help="只扫少量块大小")
ap.add_argument("--shape", default="64,2048,8,64", help="B,S,H,D")
args = ap.parse_args()
B, S, H, D = map(int, args.shape.split(","))
causal = args.causal
flops = 4 * B * H * S * S * D * (0.5 if causal else 1.0)

keys = jax.random.split(jax.random.key(0), 3)
q, k, v = (jax.random.normal(kk, (B, S, H, D), jnp.bfloat16) for kk in keys)
qh, kh, vh = (x.transpose(0, 2, 1, 3) for x in (q, k, v))  # [B, H, S, D]


@jax.jit
def reference(q, k, v):
    s = jnp.einsum("bqhd,bkhd->bhqk", q, k, preferred_element_type=jnp.float32)
    s = s / math.sqrt(D)
    if causal:
        s = jnp.where(jnp.tril(jnp.ones((S, S), bool)), s, -1e30)
    p = jax.nn.softmax(s, axis=-1).astype(v.dtype)
    return jnp.einsum("bhqk,bkhd->bqhd", p, v)


want = reference(q, k, v)
results = []


def run(name, fn, *xs, to_bshd=lambda o: o):
    try:
        out = fn(*xs)
        err = max_abs_err(to_bshd(out), want)
        ms = bench(fn, *xs, iters=args.iters)
    except Exception as e:  # 某些块大小组合不合法
        print(f"{name:58s} FAILED: {type(e).__name__}: {str(e)[:100]}")
        return
    tf = flops / ms / 1e9
    flag = "" if err <= 2e-2 else f"  <-- MISMATCH {err:.3g}"
    print(f"{name:58s} {ms:8.3f} ms  {tf:7.1f} TFLOP/s  err {err:.3g}{flag}", flush=True)
    results.append((ms, name))


def one_mask():
    return mask_lib.CausalMask((S, S)) if causal else mask_lib.FullMask((S, S))


def splash_kernel(heads, bq, bkv, bkvc):
    bs = splash.BlockSizes(block_q=bq, block_kv=bkv, block_kv_compute=bkvc)
    return splash.make_splash_mha_single_device(
        mask_lib.MultiHeadMask([one_mask()] * heads), block_sizes=bs
    )


scale = 1.0 / math.sqrt(D)

# 1) XLA 参考与 jax.nn.dot_product_attention
run("XLA einsum+softmax (fp32 softmax)", reference, q, k, v)
dpa = jax.jit(lambda q, k, v: jax.nn.dot_product_attention(
    q, k, v, is_causal=causal, implementation="xla"))
run("jax.nn.dot_product_attention(xla)", dpa, q, k, v)

# 2) splash：块大小扫描（kernel only，输入已是 [B,H,S,D]，q 已预乘 scale）
qs = (qh * scale).astype(qh.dtype)
to_bshd = lambda o: o.transpose(0, 2, 1, 3)
blocks = [128, 256, 512, 1024, 2048] if not args.quick else [512, 1024, 2048]
best = None
for bq, bkv in itertools.product(blocks, blocks):
    if bq > S or bkv > S:
        continue
    for bkvc in sorted({min(bkv, c) for c in (128, 256, 512, bkv)}):
        if bkv % bkvc:
            continue
        kern = splash_kernel(H, bq, bkv, bkvc)
        fn = jax.jit(jax.vmap(kern))
        n0 = len(results)
        run(f"splash vmap  bq={bq:4d} bkv={bkv:4d} bkv_compute={bkvc:4d}", fn,
            qs, kh, vh, to_bshd=to_bshd)
        if len(results) > n0 and (best is None or results[-1][0] < best[0]):
            best = (results[-1][0], bq, bkv, bkvc)

if best:
    _, bq, bkv, bkvc = best
    # 3) 同一最优块大小：batch 折进 head 维（一次 kernel 调用处理 B*H 个 head）
    kern_bh = splash_kernel(B * H, bq, bkv, bkvc)
    fold = jax.jit(lambda q, k, v: kern_bh(
        q.reshape(B * H, S, D), k.reshape(B * H, S, D), v.reshape(B * H, S, D)
    ).reshape(B, H, S, D))
    run(f"splash fold B*H heads bq={bq} bkv={bkv} bkvc={bkvc}", fold, qs, kh, vh,
        to_bshd=to_bshd)
    # 4) 同一最优块大小：含 scale 和 [B,S,H,D] 转置的端到端
    kern = splash_kernel(H, bq, bkv, bkvc)

    @jax.jit
    def e2e(q, k, v):
        qh = (q * scale).astype(q.dtype).transpose(0, 2, 1, 3)
        out = jax.vmap(kern)(qh, k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3))
        return out.transpose(0, 2, 1, 3)

    run(f"splash e2e [B,S,H,D] (含转置) bq={bq} bkv={bkv} bkvc={bkvc}", e2e, q, k, v)

print("\nTop 5:")
for ms, name in sorted(results)[:5]:
    print(f"  {ms:8.3f} ms  {flops / ms / 1e9:7.1f} TFLOP/s  {name}")
