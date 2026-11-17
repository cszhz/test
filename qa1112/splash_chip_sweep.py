"""Splash attention 整颗芯片（2 个 TensorCore）bf16 causal 扫描：每个 shape 选最优块大小，输出 fwd / bwd。

batch 用 shard_map 切到同一芯片的两个核；时间取 XProf 里两核中较慢者（中位数）。
bwd = (fwd+bwd) - fwd，fused bwd；时间含 q 乘 1/sqrt(dim)。

  source /zzlfs/temu/tools/sc_repro/env.sh
  /zzlfs/temu/.venv-epic/bin/python splash_chip_sweep.py \\
      [--shapes 128,1024,4,128 128,1024,2,256]
  形状格式 batch,seq,heads,dim；batch 必须是偶数。--no-causal 测 full mask。
"""

import argparse
import glob
import math
import os
import shutil

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as sk
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as smk
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

ap = argparse.ArgumentParser()
ap.add_argument("--shapes", nargs="+",
                default=["128,1024,4,128", "128,1024,2,256"])
ap.add_argument("--no-causal", action="store_true")
ap.add_argument("--iters", type=int, default=10)
ap.add_argument("--out", default="/tmp/splash-chip-sweep")
args = ap.parse_args()
causal = not args.no_causal

d0 = jax.devices()[0]
cores = [d for d in jax.devices() if d.coords == d0.coords]  # 同一芯片的两个核
mesh = Mesh(np.array(cores), ("core",))
spec = P("core")


def fwd_candidates(S):
    c = {(min(S, 1024), min(S, 1024), 512), (min(S, 1024), min(S, 2048), 512),
         (min(S, 512), min(S, 1024), 512), (min(S, 2048), min(S, 2048), 512)}
    return sorted((q, kv, min(kvc, kv)) for q, kv, kvc in c)


def bwd_candidates(S):
    return sorted({min(S, 1024), min(S, 512)})


def build(S, H, D, fwd_blk, bwd_blk):
    bq, bkv, bkvc = fwd_blk
    one = (lambda: smk.CausalMask((S, S))) if causal else (lambda: smk.FullMask((S, S)))
    kernel = sk.make_splash_mha(
        mask=smk.MultiHeadMask([one() for _ in range(H)]),
        block_sizes=sk.BlockSizes(
            block_q=bq, block_kv=bkv, block_kv_compute=bkvc,
            block_q_dkv=bwd_blk, block_kv_dkv=bwd_blk, block_kv_dkv_compute=bwd_blk,
            use_fused_bwd_kernel=True),
        head_shards=1, q_seq_shards=1)
    scale = 1.0 / math.sqrt(D)

    def attn(q, k, v):
        return jax.vmap(kernel)((q * scale).astype(q.dtype), k, v)

    def attn_fwd_bwd(q, k, v, do):
        out, vjp = jax.vjp(attn, q, k, v)
        return (out, *vjp(do))

    fwd = jax.jit(jax.shard_map(attn, mesh=mesh, in_specs=(spec,) * 3,
                                out_specs=spec, check_vma=False))
    fb = jax.jit(jax.shard_map(attn_fwd_bwd, mesh=mesh, in_specs=(spec,) * 4,
                               out_specs=(spec,) * 4, check_vma=False))
    return fwd, fb


def profile(runs, tag):
    """runs: [(key, fn, args)] -> {key: 两核较慢者的中位数 ms}"""
    out = f"{args.out}/{tag}"
    shutil.rmtree(out, ignore_errors=True)
    with jax.profiler.trace(out):
        for _, fn, xs in runs:
            for _ in range(args.iters):
                o = fn(*xs)
            jax.block_until_ready(o)
    from jax._src.profiler import ProfileData
    pb = sorted(glob.glob(f"{out}/plugins/profile/*/*.xplane.pb"), key=os.path.getmtime)[-1]
    planes = {p.name: p for p in ProfileData.from_file(pb).planes}
    per_core = []
    for dev in cores:
        p = planes[f"/device:TPU:{dev.id}"]
        per_core.append(sorted((e.start_ns, e.duration_ns) for l in p.lines
                               if l.name == "XLA Modules" for e in l.events))
    n = args.iters
    assert all(len(m) == len(runs) * n for m in per_core), [len(m) for m in per_core]
    res = {}
    for i, (key, _, _) in enumerate(runs):
        t = [max(per_core[0][j][1], per_core[1][j][1]) for j in range(i * n, (i + 1) * n)]
        res[key] = sorted(t)[n // 2] / 1e6
    return res


print(f"bf16, causal={causal}, full chip (2 cores {[d.id for d in cores]}), fused bwd, time in ms", flush=True)
print("| batch_size | seq_len | num_head | dim | fwd | bwd | best config |")
print("|---|---|---|---|---|---|---|", flush=True)
for shape in args.shapes:
    B, S, H, D = map(int, shape.split(","))
    assert B % 2 == 0, "batch must be even (split across 2 cores)"
    ks = jax.random.split(jax.random.key(0), 4)
    q, k, v, do = (jax.device_put(jax.random.normal(kk, (B, H, S, D), jnp.bfloat16),
                                  NamedSharding(mesh, spec)) for kk in ks)
    runs, fwd_done = [], set()
    for fb_blk in fwd_candidates(S):
        for bb in bwd_candidates(S):
            try:
                fwd, fb = build(S, H, D, fb_blk, bb)
                for _ in range(2):
                    jax.block_until_ready(fb(q, k, v, do))
                if fb_blk not in fwd_done:
                    for _ in range(2):
                        jax.block_until_ready(fwd(q, k, v))
                    runs.append((("fwd", fb_blk), fwd, (q, k, v)))
                    fwd_done.add(fb_blk)
                runs.append((("fb", fb_blk, bb), fb, (q, k, v, do)))
            except Exception as e:  # 块太大编译失败（VMEM）等
                print(f"<!-- skip {shape} fwd {fb_blk} bwd {bb}: {type(e).__name__} -->", flush=True)
    t = profile(runs, shape.replace(",", "x"))
    best = min((key for key in t if key[0] == "fb"), key=lambda key: t[key])
    _, fb_blk, bb = best
    f_ms = t[("fwd", fb_blk)]
    print(f"| {B} | {S} | {H} | {D} | {f_ms:.3f} | {t[best] - f_ms:.3f} | "
          f"block_q={fb_blk[0]}, block_kv={fb_blk[1]}, block_kv_compute={fb_blk[2]}, "
          f"block_dkv={bb} |", flush=True)
