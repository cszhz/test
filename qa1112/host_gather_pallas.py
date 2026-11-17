"""Embedding 表放在 host pinned 内存，TPU 直接按乱序下标 DMA 取数据（对应 CUDA 的 zero-copy gather）。

CUDA 版：GPU 线程通过 UVA 直接读 pinned host 内存，values_d[i] = values_p[indices[i]]。
TPU 版：Pallas TensorCore kernel 的输入放在 pl.HOST 内存空间（数组在 jax 里是 memory_kind="pinned_host"），
       kernel 用 DMA 引擎经 PCIe 把 table 里的数据拷到 HBM 输出，同时保持多个 DMA 在途；不经过 CPU。
       下标通过 scalar prefetch 放进 SMEM。

块粒度（TPU 实测，v6e，JAX 0.11.2 / libtpu 0.0.50）
- 偏移是运行时下标时，host 数组每行（最内维）取 4 KiB 的整数倍，一次取对齐的 8 行（一个 (8,128) tile 的高度），
  即每块 8 x 4 KiB = 32 KiB。表按"块"组织（每块 GROUP 行），按乱序块下标取块。
- SparseCore 的 Pallas lowering 不支持 host 内存，SparseCore 只能 gather HBM 里的表（适合热点部分）。

运行
  导出检查（任意机器）：PYTHONPATH=pylib JAX_PLATFORMS=cpu python host_gather_pallas.py --export-only
  TPU 上执行并校验：python host_gather_pallas.py [--blocks 16384 --group 8 --dim 1024 --n 2048]
"""

import argparse
import functools
import time

import numpy as np

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

IN_FLIGHT = 32          # 同时在途的 DMA 数


def _gather_kernel(idx_ref, table_ref, out_ref, sems, *, group):
    """idx_ref: SMEM i32[n]；table_ref: HOST [blocks*group, dim]；out_ref: HBM [n*group, dim]。

    第 i 个 DMA 把第 idx[i] 块（group 行）拷到输出的第 i 块。"""
    n = idx_ref.shape[0]

    def rows(k):
        return pl.ds(pl.multiple_of(k * group, group), group)

    def copy(i):
        return pltpu.make_async_copy(table_ref.at[rows(idx_ref[i])], out_ref.at[rows(i)], sems.at[i % IN_FLIGHT])

    @pl.loop(0, n)
    def _(i):
        @pl.when(i >= IN_FLIGHT)
        def _():
            copy(i - IN_FLIGHT).wait()       # 复用信号量前先等最早的那个完成
        copy(i).start()

    @pl.loop(jnp.maximum(n - IN_FLIGHT, 0), n)
    def _(i):
        copy(i).wait()


@functools.partial(jax.jit, static_argnames="group")
def host_gather(table_host, indices, group=8):
    n, dim = indices.shape[0], table_host.shape[1]
    return pl.pallas_call(
        functools.partial(_gather_kernel, group=group),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(1,),
            in_specs=[pl.BlockSpec(memory_space=pl.HOST)],
            out_specs=pl.BlockSpec(memory_space=pl.ANY),
            scratch_shapes=[pltpu.SemaphoreType.DMA((IN_FLIGHT,))],
        ),
        out_shape=jax.ShapeDtypeStruct((n * group, dim), table_host.dtype),
    )(indices, table_host)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=16384)
    ap.add_argument("--group", type=int, default=8)       # 每块行数（8 的倍数）
    ap.add_argument("--dim", type=int, default=1024)      # f32 x 1024 = 4 KiB / 行；块 = 32 KiB
    ap.add_argument("--n", type=int, default=2048)        # 取多少块
    ap.add_argument("--export-only", action="store_true")
    args = ap.parse_args()
    dtype = jnp.float32
    rows = args.blocks * args.group

    if args.export_only:
        from jax import export as jexport
        # 导出时用 CPU 设备的 pinned_host sharding 描述"这个参数在 host 内存"，目标平台是 TPU
        host = jax.sharding.SingleDeviceSharding(jax.devices()[0], memory_kind="pinned_host")
        table = jax.ShapeDtypeStruct((rows, args.dim), dtype, sharding=host)
        idx = jax.ShapeDtypeStruct((args.n,), jnp.int32)
        exp = jexport.export(jax.jit(functools.partial(host_gather, group=args.group)), platforms=["tpu"])(table, idx)
        text = exp.mlir_module()
        print(f"导出成功：tpu_custom_call={'tpu_custom_call' in text}，"
              f"host 内存参数={'pinned_host' in text or 'memory_space' in text}")
        return

    dev = jax.devices()[0]
    host = jax.sharding.SingleDeviceSharding(dev, memory_kind="pinned_host")
    rng = np.random.default_rng(0)
    table_np = rng.standard_normal((rows, args.dim), np.float32).astype(dtype)
    table = jax.device_put(table_np, host)                        # 表在 host pinned 内存
    idx_np = rng.integers(0, args.blocks, args.n, dtype=np.int32)  # 高度乱序
    idx = jax.device_put(idx_np, dev)

    out = jax.block_until_ready(host_gather(table, idx, group=args.group))
    want = table_np.reshape(args.blocks, args.group, args.dim)[idx_np].reshape(-1, args.dim)
    ok = np.array_equal(np.asarray(out), want)
    t0 = time.perf_counter()
    for _ in range(10):
        out = host_gather(table, idx, group=args.group)
    jax.block_until_ready(out)
    dt = (time.perf_counter() - t0) / 10
    blk = args.group * args.dim * np.dtype(dtype).itemsize
    gb = args.n * blk / 1e9
    print(f"正确={ok}；{args.n} 块 × {blk // 1024} KiB（表 {rows * args.dim * np.dtype(dtype).itemsize / 2**30:.2f} GiB）："
          f"{dt * 1e3:.2f} ms，{gb / dt:.2f} GB/s，{args.n / dt / 1e6:.3f} M 块/s")


if __name__ == "__main__":
    main()
