"""Flash Attention（MHA）在 TPU 上的实现：Pallas splash attention。

目标 shape：q/k/v = [batch, seq, head, dim] = [64, 2048, 8, 64]，bf16。

要点
- splash attention 每次处理一个样本，输入 [heads, seq, dim]（head_dim 在最内层），
  batch 维用 jax.vmap。外部常见的 [B, S, H, D] 需要先转成 [B, H, S, D]。
- kernel 不乘 softmax 缩放，调用前把 q 乘上 1/sqrt(dim)。
- mask 用 MultiHeadMask（每个 head 一个），CausalMask / FullMask / LocalMask 可选；
  splash 按 mask 的块稀疏跳过整块不需要算的部分（causal 约省一半）。
- head_dim=64 小于 TPU 的 128 lane，QK^T 的收缩维和 PV 的输出维都只用到一半，
  这是硬件限制；在这个 shape 下 attention 计算量不大，一般不是瓶颈。

运行
  TPU：python flash_attention_mha.py [--causal] [--block-q 512 --block-kv 512]
  CPU 检查：JAX_PLATFORMS=cpu python flash_attention_mha.py --small
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_kernel as splash,
)
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_mask as mask_lib,
)

from common import bench, parse_args, report


def make_mha(seq: int, heads: int, *, causal: bool, block_q: int, block_kv: int,
             interpret: bool = False):
    """构造 [B, S, H, D] -> [B, S, H, D] 的 MHA 函数。"""
    one = (mask_lib.CausalMask((seq, seq)) if causal
           else mask_lib.FullMask((seq, seq)))
    mask = mask_lib.MultiHeadMask([one] * heads)
    block_sizes = splash.BlockSizes(
        block_q=block_q, block_kv=block_kv, block_kv_compute=min(block_kv, 512)
    )
    kernel = splash.make_splash_mha_single_device(
        mask, block_sizes=block_sizes, interpret=interpret
    )

    @jax.jit
    def mha(q, k, v):
        scale = 1.0 / math.sqrt(q.shape[-1])
        # [B, S, H, D] -> [B, H, S, D]；XLA 通常把这次转置融合进前面的 QKV 投影
        qh = (q * scale).astype(q.dtype).transpose(0, 2, 1, 3)
        kh = k.transpose(0, 2, 1, 3)
        vh = v.transpose(0, 2, 1, 3)
        out = jax.vmap(kernel)(qh, kh, vh)          # [B, H, S, D]
        return out.transpose(0, 2, 1, 3)            # 回到 [B, S, H, D]

    return mha


@functools.partial(jax.jit, static_argnames="causal")
def reference(q, k, v, causal: bool):
    """纯 jnp 参考实现（fp32 softmax）。"""
    d = q.shape[-1]
    s = jnp.einsum("bqhd,bkhd->bhqk", q, k,
                   preferred_element_type=jnp.float32) / math.sqrt(d)
    if causal:
        n = q.shape[1]
        s = jnp.where(jnp.tril(jnp.ones((n, n), bool)), s, -1e30)
    p = jax.nn.softmax(s, axis=-1).astype(v.dtype)
    return jnp.einsum("bhqk,bkhd->bqhd", p, v)


def main():
    def extra(ap):
        ap.add_argument("--causal", action="store_true")
        ap.add_argument("--block-q", type=int, default=2048)
        ap.add_argument("--block-kv", type=int, default=2048)
    args = parse_args(__doc__.splitlines()[0], extra)

    if args.small:
        b, s, h, d, bq, bkv = 2, 256, 2, 64, 128, 128
    else:
        b, s, h, d, bq, bkv = 64, 2048, 8, 64, args.block_q, args.block_kv
    keys = jax.random.split(jax.random.key(0), 3)
    q, k, v = (jax.random.normal(kk, (b, s, h, d), jnp.bfloat16) for kk in keys)

    mha = make_mha(s, h, causal=args.causal, block_q=bq, block_kv=bkv,
                   interpret=args.small)
    out = mha(q, k, v)
    report(f"splash MHA [{b},{s},{h},{d}] causal={args.causal}",
           out, reference(q, k, v, args.causal), atol=2e-2)

    if not args.small:
        ms = bench(mha, q, k, v, iters=args.iters)
        flops = 4 * b * h * s * s * d * (0.5 if args.causal else 1.0)
        print(f"splash: {ms:.3f} ms/iter, {flops / ms / 1e9:.1f} TFLOP/s")
        ms_ref = bench(reference, q, k, v, args.causal, iters=args.iters)
        print(f"jnp 参考（XLA 融合）: {ms_ref:.3f} ms/iter")


if __name__ == "__main__":
    main()
