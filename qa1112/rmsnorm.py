"""RmsNorm 在 TPU 上的实现。

推荐：直接用 jnp 写（rmsnorm_jnp）。XLA 会把它和前后的 elementwise、残差加法融合成一个
kernel，手写通常不会更快。

演示：Pallas 版 fused residual-add + RmsNorm（rmsnorm_residual_pallas）。只有在需要把
它和别的东西（量化、残差、下一个 matmul 的输入准备）强行融合时才值得手写。
写法要点：按行分块（每块 block_rows 行 × 整个 hidden），hidden 维放在 lane 上
（最好是 128 的倍数），fp32 里做平方和与 rsqrt，结果再转回 bf16。

shape：tokens = 64 * 2048 = 131072，hidden = 8 * 64 = 512，bf16。

运行
  TPU：python rmsnorm.py
  CPU 检查：JAX_PLATFORMS=cpu python rmsnorm.py --small
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from common import bench, parse_args, report

EPS = 1e-6


@jax.jit
def rmsnorm_jnp(x, weight):
    xf = x.astype(jnp.float32)
    y = xf * jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + EPS)
    return (y * weight.astype(jnp.float32)).astype(x.dtype)


@jax.jit
def residual_rmsnorm_jnp(x, residual, weight):
    h = (x.astype(jnp.float32) + residual.astype(jnp.float32))
    y = h * jax.lax.rsqrt(jnp.mean(h * h, axis=-1, keepdims=True) + EPS)
    return (y * weight.astype(jnp.float32)).astype(x.dtype), h.astype(x.dtype)


def _residual_rmsnorm_kernel(x_ref, r_ref, w_ref, out_ref, new_r_ref):
    h = x_ref[...].astype(jnp.float32) + r_ref[...].astype(jnp.float32)
    inv = jax.lax.rsqrt(jnp.mean(h * h, axis=-1, keepdims=True) + EPS)
    out_ref[...] = (h * inv * w_ref[...].astype(jnp.float32)).astype(out_ref.dtype)
    new_r_ref[...] = h.astype(new_r_ref.dtype)


@functools.partial(jax.jit, static_argnames=("block_rows", "interpret"))
def residual_rmsnorm_pallas(x, residual, weight, *, block_rows=512, interpret=False):
    """返回 (rmsnorm(x + residual) * weight, x + residual)。"""
    tokens, hidden = x.shape
    assert tokens % block_rows == 0, "tokens 需是 block_rows 的倍数（否则先 pad）"
    row_spec = pl.BlockSpec((block_rows, hidden), lambda i: (i, 0))
    return pl.pallas_call(
        _residual_rmsnorm_kernel,
        grid=(tokens // block_rows,),
        in_specs=[row_spec, row_spec, pl.BlockSpec((1, hidden), lambda i: (0, 0))],
        out_specs=[row_spec, row_spec],
        out_shape=[jax.ShapeDtypeStruct(x.shape, x.dtype)] * 2,
        interpret=interpret,
    )(x, residual, weight.reshape(1, hidden))


def main():
    args = parse_args(__doc__.splitlines()[0])
    tokens, hidden, block = (1024, 512, 256) if args.small else (64 * 2048, 512, 512)
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    x = jax.random.normal(k1, (tokens, hidden), jnp.bfloat16)
    r = jax.random.normal(k2, (tokens, hidden), jnp.bfloat16)
    w = (1 + 0.1 * jax.random.normal(k3, (hidden,))).astype(jnp.bfloat16)

    want_out, want_r = residual_rmsnorm_jnp(x, r, w)
    got_out, got_r = residual_rmsnorm_pallas(x, r, w, block_rows=block,
                                             interpret=args.small)
    report("fused residual+RmsNorm 输出", got_out, want_out, atol=3e-2)
    report("fused residual+RmsNorm 新残差", got_r, want_r, atol=1e-2)

    if not args.small:
        gb = 4 * tokens * hidden * 2 / 1e9          # 读 x、r，写 out、new_r（权重可忽略）
        for name, fn, a in [
            ("jnp RmsNorm", rmsnorm_jnp, (x, w)),
            ("jnp residual+RmsNorm（XLA 融合）", residual_rmsnorm_jnp, (x, r, w)),
            ("Pallas residual+RmsNorm", functools.partial(
                residual_rmsnorm_pallas, block_rows=block), (x, r, w)),
        ]:
            ms = bench(fn, *a, iters=args.iters)
            print(f"{name}: {ms:.3f} ms/iter")
        print(f"（参考：fused 版每次约搬运 {gb:.2f} GB，按 HBM 带宽估算下限）")


if __name__ == "__main__":
    main()
