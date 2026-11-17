"""PageAttention 在 TPU 上的实现：Pallas ragged_paged_attention（vLLM TPU 后端在用）。

一次调用处理 prefill 和 decode 混在一起、长度各不相同的一批请求（continuous batching）。

数据布局（和 CUDA 版 PagedAttention 不同）
- q:           [num_tokens, num_q_heads, head_dim]，所有请求的 query 首尾拼接
- kv_pages:    [total_pages, page_size, 2 * num_kv_heads, head_dim]
               K 在偶数位（0::2），V 在奇数位（1::2）
- kv_lens:     i32[max_num_seqs]   每个请求的 KV 长度（含本次新 token）
- page_indices i32[max_num_seqs, pages_per_seq]  每个请求的页表
- cu_q_lens:   i32[max_num_seqs + 1]  query 长度的前缀和
- num_seqs:    i32[1]
causal mask 在 kernel 内部按 (kv_len - q_len) 偏移自动生成。

本例：head=8、head_dim=64（与 MHA 题目一致），一批里 4 个 prefill（各 2048 token）
+ 60 个 decode（各 1 token，上下文 2048），page_size=64。
head_dim=64 同样按 128 lane 处理，KV cache 规划 HBM 时要按实际布局算。

运行
  TPU：python paged_attention.py
  CPU 检查（只跑 jnp 参考实现，验证页表/布局构造）：JAX_PLATFORMS=cpu python paged_attention.py --small
"""

import numpy as np

import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.ragged_paged_attention import kernel as rpa

from common import bench, parse_args, report


def build_batch(prefill_lens, decode_ctx_lens, *, heads, kv_heads, dim, page_size,
                seed=0):
    """构造一批混合请求：返回 q、kv_pages 和元数据。页号随机打乱，模拟真实分配。"""
    q_lens = list(prefill_lens) + [1] * len(decode_ctx_lens)
    kv_lens = list(prefill_lens) + list(decode_ctx_lens)
    num_seqs = len(q_lens)
    pages_per_seq = max(-(-n // page_size) for n in kv_lens)
    total_pages = sum(-(-n // page_size) for n in kv_lens) + 1   # 0 号页留空
    rng = np.random.default_rng(seed)
    free = rng.permutation(np.arange(1, total_pages))
    page_indices = np.zeros((num_seqs, pages_per_seq), np.int32)
    cursor = 0
    for i, n in enumerate(kv_lens):
        need = -(-n // page_size)
        page_indices[i, :need] = free[cursor:cursor + need]
        cursor += need
    cu_q_lens = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)

    k1, k2 = jax.random.split(jax.random.key(seed))
    q = jax.random.normal(k1, (int(cu_q_lens[-1]), heads, dim), jnp.bfloat16)
    kv_pages = jax.random.normal(
        k2, (total_pages, page_size, 2 * kv_heads, dim), jnp.bfloat16)
    meta = (jnp.asarray(kv_lens, jnp.int32), jnp.asarray(page_indices),
            jnp.asarray(cu_q_lens), jnp.asarray([num_seqs], jnp.int32))
    return q, kv_pages, meta


def main():
    args = parse_args(__doc__.splitlines()[0])
    heads = kv_heads = 8
    dim = 64
    if args.small:
        prefill, decode, page_size = [96, 40], [130, 64, 7], 16
    else:
        prefill, decode, page_size = [2048] * 4, [2048] * 60, 64
    q, kv_pages, (kv_lens, page_indices, cu_q_lens, num_seqs) = build_batch(
        prefill, decode, heads=heads, kv_heads=kv_heads, dim=dim,
        page_size=page_size)
    sm_scale = dim ** -0.5
    want = rpa.ref_ragged_paged_attention(
        q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs, sm_scale=sm_scale)

    if args.small:
        # 该 kernel 依赖 TPU 的 scalar prefetch / DMA，CPU 上只验证参考实现与布局。
        print(f"[OK] 参考实现可运行：q {q.shape}, kv_pages {kv_pages.shape}, "
              f"out {want.shape}; TPU 上会与 Pallas kernel 对比")
        return

    # kernel 要求 head_dim 是 128 的倍数（l/m 统计量按 128 lane 存放，再广播到 head_dim）。
    # head_dim=64 时把 q 和 kv_pages 的最后一维补零到 128；sm_scale 仍按真实 dim，输出截回前 dim 维。
    # 补的是 0：QKᵀ 不变，PV 多出的列全是 0，结果与不补齐在数学上相同。
    pad = (-dim) % 128
    if pad:
        q_in = jnp.pad(q, ((0, 0), (0, 0), (0, pad)))
        kv_in = jnp.pad(kv_pages, ((0, 0), (0, 0), (0, 0), (0, pad)))
        fn = jax.jit(lambda q_, kv_, *a: rpa.ragged_paged_attention(q_, kv_, *a, sm_scale=sm_scale)[..., :dim])
    else:
        q_in, kv_in = q, kv_pages
        fn = jax.jit(lambda *a: rpa.ragged_paged_attention(*a, sm_scale=sm_scale))
    q, kv_pages = q_in, kv_in
    got = fn(q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs)
    report(f"ragged_paged_attention {len(prefill)} prefill + {len(decode)} decode",
           got, want, atol=2e-2)
    ms = bench(fn, q, kv_pages, kv_lens, page_indices, cu_q_lens, num_seqs,
               iters=args.iters)
    print(f"ragged_paged_attention: {ms:.3f} ms/iter（{q.shape[0]} 个 query token）")


if __name__ == "__main__":
    main()
