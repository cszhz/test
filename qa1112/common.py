"""公共工具：计时、误差比较、命令行参数。"""

import argparse
import time

import jax
import jax.numpy as jnp


def parse_args(description: str, extra=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument(
        "--small",
        action="store_true",
        help="小 shape + Pallas interpret 模式，在 CPU 上检查正确性（JAX_PLATFORMS=cpu）",
    )
    ap.add_argument("--iters", type=int, default=20, help="计时迭代次数")
    if extra is not None:
        extra(ap)
    return ap.parse_args()


def bench(fn, *args, iters: int = 20, warmup: int = 3) -> float:
    """返回每次调用的平均耗时（毫秒），包含 block_until_ready。"""
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / iters * 1e3


def max_abs_err(a, b) -> float:
    return float(jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32))))


def report(name: str, got, want, atol: float) -> None:
    err = max_abs_err(got, want)
    status = "OK" if err <= atol else "MISMATCH"
    print(f"[{status}] {name}: max|err| = {err:.4g} (atol {atol})")
    if err > atol:
        raise SystemExit(1)
