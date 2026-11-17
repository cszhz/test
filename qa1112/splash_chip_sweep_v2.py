"""Splash attention 整颗芯片（2 个 TensorCore）块大小寻优，v2（splash_chip_sweep.py 的优化版）。

和 v1 的区别：
  * fwd / bwd 分开搜索：fwd 和 bwd 是两个独立的 kernel，总时间 = fwd(块_f) + bwd(块_b)，
    所以先只测 fwd 选出最优块，再固定 fwd 搜 bwd，组合数从 F×B 降到 F+B，省下的预算用来扩大网格：
      fwd：block_q × block_kv × block_kv_compute（可选 K/V 布局 HEAD_DIM_MINOR / SEQ_MINOR）
      bwd：block_q_dkv × block_kv_dkv × block_kv_dkv_compute（fused），可选 --unfused（再加 dq 块）
  * 两轮淘汰：第一轮每个候选 --quick-iters 次，保留前 --keep（比例）且不少于 --min-keep 个，第二轮 --iters 次。
  * 候选先做合法性过滤（128 的倍数、整除 seq、compute 整除 block），再多线程 AOT 编译（--threads）。
  * 每个候选的 jit 函数名唯一（c<编号>），按名字从 XPlane 取时间，不依赖事件顺序；时间取两核中较慢者的中位数。
  * 最优配置和纯 JAX 参考实现比对 fwd 输出和 dq/dk/dv（--no-check 跳过）。
  * 输出 markdown 表（含 TFLOP/s）和 JSON（所有测量，--json），--resume 跳过 JSON 里已有的 shape。
  * 没有 TPU 时（JAX_PLATFORMS=cpu）用 Pallas interpret 模式 + 墙钟计时，只用来检查脚本逻辑。

  source /zzlfs/temu/tools/sc_repro/env.sh
  /zzlfs/temu/.venv-epic/bin/python splash_chip_sweep_v2.py \\
      [--shapes 128,1024,4,128 128,1024,2,256] [--layouts] [--unfused] [--json /tmp/splash.json]
  形状格式 batch,seq,heads,dim；batch 必须能被芯片核数（2）整除。--no-causal 测 full mask。

  CPU 逻辑检查（很慢，只用小 shape）：
  JAX_PLATFORMS=cpu python splash_chip_sweep_v2.py --shapes 2,256,1,128 --iters 2 --quick-iters 1
"""

import argparse
import concurrent.futures as cf
import glob
import itertools
import json
import math
import os
import re
import shutil
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as sk
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as smk
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

ap = argparse.ArgumentParser()
ap.add_argument("--shapes", nargs="+", default=["128,1024,4,128", "128,1024,2,256"])
ap.add_argument("--no-causal", action="store_true")
ap.add_argument("--iters", type=int, default=10, help="第二轮（精测）每个候选的次数")
ap.add_argument("--quick-iters", type=int, default=3, help="第一轮（粗筛）每个候选的次数")
ap.add_argument("--keep", type=float, default=0.25, help="第一轮保留的比例")
ap.add_argument("--min-keep", type=int, default=4)
ap.add_argument("--blocks", default="128,256,512,1024,2048", help="块大小候选（≤ seq 的才用）")
ap.add_argument("--max-kv-block", type=int, default=4096, help="block_kv 额外允许到这么大")
ap.add_argument("--layouts", action="store_true", help="也扫 K/V 的 SEQ_MINOR 布局")
ap.add_argument("--unfused", action="store_true", help="也扫非 fused bwd（dkv + dq 两个 kernel）")
ap.add_argument("--threads", type=int, default=8, help="并行 AOT 编译的线程数")
ap.add_argument("--no-check", action="store_true", help="跳过和参考实现的数值比对")
ap.add_argument("--json", default="/tmp/splash-chip-sweep-v2.json")
ap.add_argument("--resume", action="store_true")
ap.add_argument("--out", default="/tmp/splash-chip-sweep-v2", help="XPlane trace 目录")
args = ap.parse_args()
causal = not args.no_causal

d0 = jax.devices()[0]
ON_TPU = d0.platform == "tpu"
if ON_TPU:
    cores = [d for d in jax.devices() if d.coords == d0.coords]  # 同一芯片的两个核
else:
    cores = [d0]
mesh = Mesh(np.array(cores), ("core",))
spec = P("core")
LAYOUTS = [sk.QKVLayout.HEAD_DIM_MINOR] + ([sk.QKVLayout.SEQ_MINOR] if args.layouts else [])


# ----------------------------------------------------------------------------- candidates
def _sizes(S, extra_max=None):
    base = [int(b) for b in args.blocks.split(",")]
    if extra_max:
        b = max(base)
        while b * 2 <= extra_max:
            b *= 2
            base.append(b)
    return sorted({b for b in base if b <= S and S % b == 0 and b % 128 == 0})


def fwd_candidates(S):
    out = []
    for bq, bkv, layout in itertools.product(_sizes(S), _sizes(S, args.max_kv_block), LAYOUTS):
        for bkvc in _sizes(S):
            if bkvc <= bkv and bkv % bkvc == 0:
                out.append(dict(block_q=bq, block_kv=bkv, block_kv_compute=bkvc, kv_layout=layout.name))
    return out


def bwd_candidates(S):
    out = []
    for bq, bkv in itertools.product(_sizes(S), _sizes(S, args.max_kv_block)):
        for bkvc in _sizes(S):
            if bkvc <= bkv and bkv % bkvc == 0:
                out.append(dict(fused=True, block_q_dkv=bq, block_kv_dkv=bkv, block_kv_dkv_compute=bkvc))
    if args.unfused:
        for bq, bkv, bkvc in itertools.product(_sizes(S), _sizes(S), _sizes(S)):
            if bkvc <= bkv and bkv % bkvc == 0:
                for bqdq, bkvdq in itertools.product(_sizes(S), _sizes(S)):
                    out.append(dict(fused=False, block_q_dkv=bq, block_kv_dkv=bkv,
                                    block_kv_dkv_compute=bkvc, block_q_dq=bqdq, block_kv_dq=bkvdq))
    return out


# ----------------------------------------------------------------------------- build
def make_kernel(S, H, fwd, bwd):
    layout = sk.QKVLayout[fwd["kv_layout"]]
    kw = dict(block_q=fwd["block_q"], block_kv=fwd["block_kv"], block_kv_compute=fwd["block_kv_compute"],
              k_layout=layout, v_layout=layout)
    if bwd is not None:
        kw.update(block_q_dkv=bwd["block_q_dkv"], block_kv_dkv=bwd["block_kv_dkv"],
                  block_kv_dkv_compute=bwd["block_kv_dkv_compute"], use_fused_bwd_kernel=bwd["fused"])
        if not bwd["fused"]:
            kw.update(block_q_dq=bwd["block_q_dq"], block_kv_dq=bwd["block_kv_dq"])
    one = (lambda: smk.CausalMask((S, S))) if causal else (lambda: smk.FullMask((S, S)))
    return sk.make_splash_mha(
        mask=smk.MultiHeadMask([one() for _ in range(H)]),
        block_sizes=sk.BlockSizes(**kw), head_shards=1, q_seq_shards=1, interpret=not ON_TPU)


def attn_fn(kernel, D):
    scale = 1.0 / math.sqrt(D)

    def attn(q, k, v):
        return jax.vmap(kernel)((q * scale).astype(q.dtype), k, v)
    return attn


def compile_candidate(tag, S, H, D, fwd, bwd, xs):
    """AOT 编译一个候选，返回 (tag, compiled) 或 (tag, Exception)。函数名 = tag，用来在 XPlane 里认领时间。"""
    try:
        attn = attn_fn(make_kernel(S, H, fwd, bwd), D)
        if bwd is None:
            def f(q, k, v):
                return attn(q, k, v)
            n_in, out_specs = 3, spec
        else:
            def f(q, k, v, do):
                out, vjp = jax.vjp(attn, q, k, v)
                return (out, *vjp(do))
            n_in, out_specs = 4, (spec,) * 4
        f.__name__ = f.__qualname__ = tag
        sm = jax.shard_map(f, mesh=mesh, in_specs=(spec,) * n_in, out_specs=out_specs, check_vma=False)
        sm.__name__ = tag
        return tag, jax.jit(sm).lower(*xs[:n_in]).compile()
    except Exception as e:  # 块太大编译失败（VMEM）、不支持的组合等
        return tag, e


def compile_all(items, S, H, D, xs):
    """items: [(tag, fwd, bwd)] -> {tag: compiled}，失败的打印后丢弃。"""
    good = {}
    with cf.ThreadPoolExecutor(max_workers=args.threads) as ex:
        futs = [ex.submit(compile_candidate, tag, S, H, D, f, b, xs) for tag, f, b in items]
        for fut in cf.as_completed(futs):
            tag, res = fut.result()
            if isinstance(res, Exception):
                print(f"<!-- skip {tag}: {type(res).__name__}: {str(res).splitlines()[0][:120] if str(res) else ''} -->",
                      flush=True)
            else:
                good[tag] = res
    return good


# ----------------------------------------------------------------------------- timing
def measure(compiled, xs_for, iters, trace_tag):
    """compiled: {tag: fn}；xs_for(tag) -> 参数。返回 {tag: ms}（两核较慢者的中位数）。"""
    for tag, fn in compiled.items():  # 预热
        jax.block_until_ready(fn(*xs_for(tag)))
    if not ON_TPU:
        res = {}
        for tag, fn in compiled.items():
            ts = []
            for _ in range(iters):
                t0 = time.perf_counter()
                jax.block_until_ready(fn(*xs_for(tag)))
                ts.append((time.perf_counter() - t0) * 1e3)
            res[tag] = sorted(ts)[len(ts) // 2]
        return res
    out = f"{args.out}/{trace_tag}"
    shutil.rmtree(out, ignore_errors=True)
    with jax.profiler.trace(out):
        for tag, fn in compiled.items():
            o = None
            for _ in range(iters):
                o = fn(*xs_for(tag))
            jax.block_until_ready(o)
    from jax._src.profiler import ProfileData
    pb = sorted(glob.glob(f"{out}/plugins/profile/*/*.xplane.pb"), key=os.path.getmtime)[-1]
    planes = {p.name: p for p in ProfileData.from_file(pb).planes}
    pat = re.compile(r"(c\d+[fb])")
    per_core = []
    for dev in cores:
        d = {}
        for line in planes[f"/device:TPU:{dev.id}"].lines:
            if line.name != "XLA Modules":
                continue
            for e in line.events:
                m = pat.search(e.name)
                if m:
                    d.setdefault(m.group(1), []).append(e.duration_ns)
        per_core.append(d)
    res = {}
    for tag in compiled:
        runs = [c.get(tag, []) for c in per_core]
        n = min(len(r) for r in runs)
        if n == 0:
            print(f"<!-- no xplane events for {tag} -->", flush=True)
            continue
        t = [max(r[j] for r in runs) for j in range(n)]
        res[tag] = sorted(t)[n // 2] / 1e6
    return res


def halving(compiled, xs_for, trace_tag, always_keep=()):
    """两轮淘汰：quick 全量粗筛 -> 保留前 keep（always_keep 一定保留）-> iters 精测。
    返回 ({tag: ms 精测}, {tag: ms 粗测})。"""
    quick = measure(compiled, xs_for, args.quick_iters, trace_tag + "-quick")
    ranked = [t for t in sorted(quick, key=quick.get) if t not in always_keep]
    n_keep = max(args.min_keep, math.ceil(len(ranked) * args.keep))
    finals = {t: compiled[t] for t in list(always_keep) + ranked[:n_keep] if t in compiled}
    return measure(finals, xs_for, args.iters, trace_tag + "-final"), quick


# ----------------------------------------------------------------------------- reference check
def reference(q, k, v):
    """纯 JAX fp32 注意力（同样先乘 1/sqrt(D) 再 cast 回 bf16）。"""
    D = q.shape[-1]
    qs = (q * (1.0 / math.sqrt(D))).astype(q.dtype).astype(jnp.float32)
    s = jnp.einsum("bhqd,bhkd->bhqk", qs, k.astype(jnp.float32))
    if causal:
        S = q.shape[2]
        s = jnp.where(jnp.tril(jnp.ones((S, S), bool)), s, -1e30)
    return jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(s, -1), v.astype(jnp.float32))


def check(S, H, D, fwd, bwd, q, k, v, do):
    """在一个核的 2 条 batch 上比对 fwd 输出和 dq/dk/dv，返回最大相对误差。"""
    n = 2
    q, k, v, do = (np.asarray(x[:n]) for x in (q, k, v, do))
    attn = attn_fn(make_kernel(S, H, fwd, bwd), D)
    out, vjp = jax.vjp(attn, q, k, v)
    grads = vjp(do)
    rout, rvjp = jax.vjp(reference, q, k, v)
    rgrads = rvjp(do.astype(jnp.float32))
    errs = {}
    for name, a, b in [("out", out, rout), ("dq", grads[0], rgrads[0]),
                       ("dk", grads[1], rgrads[1]), ("dv", grads[2], rgrads[2])]:
        a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
        errs[name] = float(np.abs(a - b).max() / (np.abs(b).max() + 1e-6))
    return errs


# ----------------------------------------------------------------------------- main
def flops(B, S, H, D):
    f = 4.0 * B * H * S * S * D * (0.5 if causal else 1.0)
    return f, 2.5 * f  # fwd，bwd（dq + dk + dv + 重算 p）


results = {}
if args.resume and os.path.exists(args.json):
    results = json.load(open(args.json))

print(f"bf16, causal={causal}, {'full chip' if ON_TPU else 'CPU interpret'} ({len(cores)} cores "
      f"{[d.id for d in cores]}), time in ms", flush=True)
print("| batch_size | seq_len | num_head | dim | fwd | bwd | fwd TFLOP/s | bwd TFLOP/s | best fwd | best bwd | check |")
print("|---|---|---|---|---|---|---|---|---|---|---|", flush=True)
for shape in args.shapes:
    if shape in results:
        r = results[shape]
        print(f"| {shape.replace(',', ' | ')} | {r['fwd_ms']:.3f} | {r['bwd_ms']:.3f} | (resumed) |||||", flush=True)
        continue
    B, S, H, D = map(int, shape.split(","))
    assert B % len(cores) == 0, f"batch must be divisible by {len(cores)} cores"
    ks = jax.random.split(jax.random.key(0), 4)
    q, k, v, do = (jax.device_put(jax.random.normal(kk, (B, H, S, D), jnp.bfloat16), NamedSharding(mesh, spec))
                   for kk in ks)
    xs = (q, k, v, do)
    t0 = time.time()

    # 1) fwd
    fcands = fwd_candidates(S)
    ftag = {f"c{i}f": c for i, c in enumerate(fcands)}
    fcomp = compile_all([(t, c, None) for t, c in ftag.items()], S, H, D, xs)
    ft, fquick = halving(fcomp, lambda t: xs[:3], shape.replace(",", "x") + "-fwd")
    best_ft = min(ft, key=ft.get)
    best_f = ftag[best_ft]

    # 2) bwd：固定最优 fwd。同一 trace 里一起测最优 fwd，bwd = fb - fwd。
    bcands = bwd_candidates(S)
    btag = {f"c{i}b": c for i, c in enumerate(bcands)}
    bcomp = compile_all([(t, best_f, c) for t, c in btag.items()], S, H, D, xs)
    bcomp_with_f = dict(bcomp)
    bcomp_with_f[best_ft] = fcomp[best_ft]
    xs_for = lambda t: xs[:3] if t.endswith("f") else xs
    bt, bquick = halving(bcomp_with_f, xs_for, shape.replace(",", "x") + "-bwd", always_keep=(best_ft,))
    f_ms = bt.pop(best_ft, ft[best_ft])
    best_bt = min(bt, key=bt.get)
    best_b = btag[best_bt]
    b_ms = bt[best_bt] - f_ms

    errs = None if args.no_check else check(S, H, D, best_f, best_b, *xs)
    ok = "skip" if errs is None else ("ok" if max(errs.values()) < 2e-2 else
                                      "FAIL " + ",".join(f"{k}={v:.2e}" for k, v in errs.items()))
    ff, bf = flops(B, S, H, D)
    nchips = 1
    print(f"| {B} | {S} | {H} | {D} | {f_ms:.3f} | {b_ms:.3f} | {ff / f_ms / 1e9 / nchips:.1f} | "
          f"{bf / max(b_ms, 1e-9) / 1e9 / nchips:.1f} | "
          f"q={best_f['block_q']},kv={best_f['block_kv']},kvc={best_f['block_kv_compute']}"
          f"{',' + best_f['kv_layout'] if args.layouts else ''} | "
          f"{'fused' if best_b['fused'] else 'unfused'},q={best_b['block_q_dkv']},kv={best_b['block_kv_dkv']},"
          f"kvc={best_b['block_kv_dkv_compute']}"
          f"{'' if best_b['fused'] else ',dq=' + str(best_b['block_q_dq']) + 'x' + str(best_b['block_kv_dq'])} "
          f"| {ok} |", flush=True)
    print(f"<!-- {shape}: fwd {len(fcands)} cands ({len(fcomp)} compiled), bwd {len(bcands)} cands "
          f"({len(bcomp)} compiled), {time.time() - t0:.0f}s -->", flush=True)

    results[shape] = dict(
        causal=causal, fwd_ms=f_ms, bwd_ms=b_ms, best_fwd=best_f, best_bwd=best_b, check=errs,
        fwd_final={t: dict(ms=v, **ftag[t]) for t, v in ft.items()},
        fwd_all_quick={t: dict(ms=v, **ftag[t]) for t, v in fquick.items()},
        bwd_all_quick={t: dict(ms=v - f_ms, **btag[t]) for t, v in bquick.items() if t in btag},
        bwd_final={t: dict(ms=v - f_ms, **btag[t]) for t, v in bt.items()},
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    json.dump(results, open(args.json, "w"), indent=1, default=str)

print(f"<!-- results: {args.json} -->")
