"""MoE（对应 SonicMoE 的核心路径）在 TPU 上的实现：top-k 路由 + 分组 GEMM。

流程（单设备，不含专家并行）
1. router：logits = x @ W_router，softmax 后取 top-k，并把 k 个权重重新归一化。
2. 排序：把 T*k 个 (token, expert) 对按 expert 稳定排序，得到连续分组；
   group_sizes[e] = 分给专家 e 的行数。
3. 分组 GEMM：h = silu(xs @ W1[e]) * (xs @ W3[e])；y = h @ W2[e]。
   - --impl gmm：Pallas megablox.gmm（MaxText 在用），tiling=(tm, tk, tn)，取 128 的倍数；
   - --impl ragged：XLA 原生 jax.lax.ragged_dot。
4. 合并：y 乘上路由权重，按 token 下标 scatter-add 回 [T, H]。

和 SonicMoE 的对应：分组 GEMM ≈ megablox gmm / ragged_dot；token 按 tile 取整 ≈ 让每组行数
和 tm 对齐（gmm 内部按 tile 处理不齐的组边界）；专家并行的 dispatch/combine 在多卡上用
jax.lax.ragged_all_to_all（本脚本未包含，见 README）。

默认 shape：tokens=8192、hidden=2048、intermediate=1024、experts=64、top-k=8，bf16。

运行
  TPU：python moe_grouped_gemm.py [--impl gmm|ragged] [--tokens 8192 ...]
  CPU 检查：JAX_PLATFORMS=cpu python moe_grouped_gemm.py --small
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.megablox import gmm as megablox_gmm

from common import bench, parse_args, report


def route(x, router_w, top_k):
    logits = jnp.dot(x, router_w, preferred_element_type=jnp.float32)
    probs = jax.nn.softmax(logits, axis=-1)
    weights, experts = jax.lax.top_k(probs, top_k)          # [T, k]
    weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
    return weights, experts


@functools.partial(jax.jit, static_argnames=("top_k", "impl", "tiling", "interpret"))
def moe(x, router_w, w1, w3, w2, *, top_k, impl="gmm", tiling=(512, 512, 512),
        interpret=False):
    tokens, hidden = x.shape
    num_experts = w1.shape[0]
    weights, experts = route(x, router_w, top_k)

    flat_experts = experts.reshape(-1)                       # [T*k]
    order = jnp.argsort(flat_experts, stable=True)           # 按专家排序
    token_idx = order // top_k
    xs = x[token_idx]                                        # [T*k, H] gather
    group_sizes = jnp.bincount(flat_experts, length=num_experts).astype(jnp.int32)

    if impl == "gmm":
        gm = functools.partial(megablox_gmm, group_sizes=group_sizes,
                               preferred_element_type=jnp.float32,
                               tiling=tiling, interpret=interpret)
        h = jax.nn.silu(gm(xs, w1)) * gm(xs, w3)
        y = gm(h.astype(x.dtype), w2)
    else:
        rd = functools.partial(jax.lax.ragged_dot, group_sizes=group_sizes,
                               preferred_element_type=jnp.float32)
        h = jax.nn.silu(rd(xs, w1)) * rd(xs, w3)
        y = rd(h.astype(x.dtype), w2)

    y = y * weights.reshape(-1)[order][:, None]
    out = jnp.zeros((tokens, hidden), jnp.float32).at[token_idx].add(y)
    return out.astype(x.dtype)


@functools.partial(jax.jit, static_argnames="top_k")
def reference(x, router_w, w1, w3, w2, *, top_k):
    """逐专家稠密计算（只用于小 shape 校验）。"""
    weights, experts = route(x, router_w, top_k)
    gate = jnp.zeros((x.shape[0], w1.shape[0]), jnp.float32)
    gate = gate.at[jnp.arange(x.shape[0])[:, None], experts].set(weights)
    h = jax.nn.silu(jnp.einsum("th,ehi->eti", x, w1, preferred_element_type=jnp.float32))
    h = h * jnp.einsum("th,ehi->eti", x, w3, preferred_element_type=jnp.float32)
    y = jnp.einsum("eti,eih->eth", h.astype(x.dtype), w2,
                   preferred_element_type=jnp.float32)
    return jnp.einsum("eth,te->th", y, gate).astype(x.dtype)


def main():
    def extra(ap):
        ap.add_argument("--impl", choices=["gmm", "ragged"], default="gmm")
        ap.add_argument("--tokens", type=int, default=8192)
        ap.add_argument("--hidden", type=int, default=2048)
        ap.add_argument("--inter", type=int, default=1024)
        ap.add_argument("--experts", type=int, default=64)
        ap.add_argument("--top-k", type=int, default=8)
        ap.add_argument("--tile", type=int, nargs=3, default=[512, 512, 512])
    args = parse_args(__doc__.splitlines()[0], extra)
    if args.small:
        t, h, i, e, k, tiling = 256, 256, 128, 8, 2, (128, 128, 128)
    else:
        t, h, i, e, k = args.tokens, args.hidden, args.inter, args.experts, args.top_k
        tiling = tuple(args.tile)

    ks = jax.random.split(jax.random.key(0), 5)
    x = jax.random.normal(ks[0], (t, h), jnp.bfloat16)
    router_w = (jax.random.normal(ks[1], (h, e)) / h ** 0.5).astype(jnp.bfloat16)
    w1 = (jax.random.normal(ks[2], (e, h, i)) / h ** 0.5).astype(jnp.bfloat16)
    w3 = (jax.random.normal(ks[3], (e, h, i)) / h ** 0.5).astype(jnp.bfloat16)
    w2 = (jax.random.normal(ks[4], (e, i, h)) / i ** 0.5).astype(jnp.bfloat16)

    run = functools.partial(moe, top_k=k, impl=args.impl, tiling=tiling,
                            interpret=args.small)
    got = run(x, router_w, w1, w3, w2)
    if args.small or t * e <= 8192 * 64:
        report(f"MoE {args.impl} T={t} H={h} I={i} E={e} k={k}",
               got, reference(x, router_w, w1, w3, w2, top_k=k), atol=5e-2)
    if not args.small:
        ms = bench(run, x, router_w, w1, w3, w2, iters=args.iters)
        flops = 2 * t * k * h * i * 3                         # W1、W3、W2 三个 GEMM
        print(f"MoE {args.impl}: {ms:.3f} ms/iter, {flops / ms / 1e9:.1f} TFLOP/s"
              f"（只计专家 GEMM）")


if __name__ == "__main__":
    main()
