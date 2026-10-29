# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
MXFP8 x MXFP4 variants of the grouped expert matmul and the fused expert matmul + EP->DP combine.

Routing, masking and the per-rank scatter follow fused_exp_matmul_ep_to_dp.py; only the operand loads
and the dot change (tl.dot_scaled):

  activations  x  (rows, K)        float8_e4m3fn, one e8m0 scale per 32 along K: xs (rows, K // 32) uint8
  weights      w  (E, K // 2, N)   e2m1 packed two per byte along K (low nibble first),
                                    one e8m0 scale per 32 along K:              ws (E, N, K // 32) uint8
  output          (rows, N)        bfloat16

Both kernels share _mx_tile, so the fused and unfused paths compute bit-identical tiles.
K and N must be multiples of the 128-wide tiles.
"""

import torch
import triton
import triton.language as tl

import iris

BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 128
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


@triton.jit
def _mx_tile(
    x_ptr,
    xs_ptr,
    w_ptr,
    ws_ptr,
    expert,
    offs_m,
    mask_m,
    offs_n,
    mask_n,
    K,
    x_stride_m,
    xs_stride_m,
    w_stride_e,
    w_stride_k,
    ws_stride_e,
    ws_stride_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_k = tl.arange(0, BLOCK_K)
    offs_kp = tl.arange(0, BLOCK_K // 2)
    offs_ks = tl.arange(0, BLOCK_K // 32)
    x_ptrs = x_ptr + offs_m[:, None] * x_stride_m + offs_k[None, :]
    xs_ptrs = xs_ptr + offs_m[:, None] * xs_stride_m + offs_ks[None, :]
    w_ptrs = w_ptr + expert * w_stride_e + offs_kp[:, None] * w_stride_k + offs_n[None, :]
    ws_ptrs = ws_ptr + expert * ws_stride_e + offs_n[:, None] * ws_stride_n + offs_ks[None, :]
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        x = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0)
        xs = tl.load(xs_ptrs, mask=mask_m[:, None], other=127)
        w = tl.load(w_ptrs, mask=mask_n[None, :], other=0)
        ws = tl.load(ws_ptrs, mask=mask_n[:, None], other=127)
        acc = tl.dot_scaled(x, xs, "e4m3", w, ws, "e2m1", acc)
        x_ptrs += BLOCK_K
        xs_ptrs += BLOCK_K // 32
        w_ptrs += (BLOCK_K // 2) * w_stride_k
        ws_ptrs += BLOCK_K // 32
    return acc


@triton.jit
def _grouped_matmul_mx_kernel(
    x_ptr,
    xs_ptr,
    w_ptr,
    ws_ptr,
    y_ptr,
    slice_offs_ptr,
    slice_sizes_ptr,
    n_experts,
    K,
    N,
    x_stride_m,
    xs_stride_m,
    w_stride_e,
    w_stride_k,
    ws_stride_e,
    ws_stride_n,
    y_stride_m,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    n_n_tiles = tl.cdiv(N, BLOCK_N)
    expert = (pid // n_n_tiles).to(tl.int64)
    pid_n = pid % n_n_tiles
    if expert >= n_experts:
        return
    slice_off = tl.load(slice_offs_ptr + expert).to(tl.int64)
    slice_size = tl.load(slice_sizes_ptr + expert)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    for pid_m in range(0, tl.cdiv(slice_size, BLOCK_M)):
        offs_m_local = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_m = slice_off + offs_m_local
        mask_m = offs_m_local < slice_size
        acc = _mx_tile(
            x_ptr, xs_ptr, w_ptr, ws_ptr, expert, offs_m, mask_m, offs_n, mask_n, K,
            x_stride_m, xs_stride_m, w_stride_e, w_stride_k, ws_stride_e, ws_stride_n,
            BLOCK_M, BLOCK_N, BLOCK_K,
        )  # fmt: skip
        y_ptrs = y_ptr + offs_m[:, None] * y_stride_m + offs_n[None, :]
        tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _fused_exp_matmul_ep_to_dp_mx_kernel(
    dst_ptr,
    dst_stride_m,
    x_ptr,
    xs_ptr,
    w_ptr,
    ws_ptr,
    slice_offs_ptr,
    slice_sizes_ptr,
    expt_filter_ptr,
    expt_filter_stride_m,
    expt_indx_ptr,
    topk_indx_ptr,
    n_local_experts,
    n_slots_per_rank,
    K,
    N,
    x_stride_m,
    xs_stride_m,
    w_stride_e,
    w_stride_k,
    ws_stride_e,
    ws_stride_n,
    heap_bases,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SRC_RANK: tl.constexpr,
    N_RANKS: tl.constexpr,
):
    pid = tl.program_id(0)
    n_n_tiles = tl.cdiv(N, BLOCK_N)
    local_expert_id = pid // n_n_tiles
    pid_n = pid % n_n_tiles
    if local_expert_id >= n_local_experts:
        return
    expert = local_expert_id.to(tl.int64)
    slice_off = tl.load(slice_offs_ptr + expert).to(tl.int64)
    slice_size = tl.load(slice_sizes_ptr + expert)
    if slice_size == 0:
        return
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)
    mask_n = offs_n < N
    for pid_m in range(0, tl.cdiv(slice_size, BLOCK_M)):
        offs_m_local = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_m = slice_off + offs_m_local
        mask_m = offs_m_local < slice_size

        # Scatter metadata for this M-tile, as in fused_exp_matmul_ep_to_dp.py.
        dst_indx_globals = tl.load(topk_indx_ptr + offs_m, mask=mask_m, other=-1)
        valid_dst = mask_m & (dst_indx_globals >= 0)
        safe_dst_indx = tl.where(valid_dst, dst_indx_globals, tl.zeros_like(dst_indx_globals))
        dst_expt_indxs = tl.load(expt_indx_ptr + safe_dst_indx, mask=valid_dst, other=0).to(tl.int32)
        expt_filter_ptr_local = expt_filter_ptr + SRC_RANK * expt_filter_stride_m
        has_dst_expts = (
            (tl.load(expt_filter_ptr_local + dst_expt_indxs // 32, mask=valid_dst, other=0) >> (dst_expt_indxs % 32))
            & 1
        ).to(tl.int1)
        row_valid = valid_dst & has_dst_expts
        dst_ranks = dst_indx_globals // n_slots_per_rank
        dst_indx_locals = dst_indx_globals - dst_ranks * n_slots_per_rank
        dst_indx_locals = tl.where(row_valid, dst_indx_locals, tl.zeros_like(dst_indx_locals))

        acc = _mx_tile(
            x_ptr, xs_ptr, w_ptr, ws_ptr, expert, offs_m, mask_m, offs_n, mask_n, K,
            x_stride_m, xs_stride_m, w_stride_e, w_stride_k, ws_stride_e, ws_stride_n,
            BLOCK_M, BLOCK_N, BLOCK_K,
        )  # fmt: skip
        out = acc.to(dst_ptr.dtype.element_ty)

        # Per-rank 2D masked scatter.
        dst_ptrs_2d = dst_ptr + dst_indx_locals[:, None] * dst_stride_m + offs_n[None, :]
        for r in tl.static_range(N_RANKS):
            rank_mask = row_valid & (dst_ranks == r)
            store_mask = rank_mask[:, None] & mask_n[None, :]
            if r == SRC_RANK:
                tl.store(dst_ptrs_2d, out, mask=store_mask)
            else:
                iris.store(dst_ptrs_2d, out, SRC_RANK, r, heap_bases, mask=store_mask, hint=(1, 16))


def quantize_mxfp8(x: torch.Tensor):
    """bf16 (rows, K) -> float8_e4m3fn values and e8m0 scales (rows, K // 32), one scale per 32 along K."""
    rows, K = x.shape
    xr = x.float().view(rows, K // 32, 32)
    amax = xr.abs().amax(-1, keepdim=True).clamp_min(2.0**-120)
    e = torch.ceil(torch.log2(amax / 448.0))
    q = (xr / torch.exp2(e)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(rows, K)
    s = (e.squeeze(-1) + 127).clamp(0, 254).to(torch.uint8).contiguous()
    return q.contiguous(), s


def random_mxfp4_weights(n_experts: int, K: int, N: int, device, generator=None, scale_exp: int = -4):
    """Random e2m1 codes packed along K, shape (E, K // 2, N), with a constant 2**scale_exp e8m0 scale."""
    w = torch.randint(0, 256, (n_experts, K // 2, N), dtype=torch.uint8, device=device, generator=generator)
    ws = torch.full((n_experts, N, K // 32), 127 + scale_exp, dtype=torch.uint8, device=device)
    return w.contiguous(), ws


def dequant_mxfp8(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    rows, K = q.shape
    return (q.float().view(rows, K // 32, 32) * torch.exp2(s.float() - 127)[..., None]).view(rows, K)


def dequant_mxfp4(w: torch.Tensor, ws: torch.Tensor) -> torch.Tensor:
    """One expert: (K // 2, N) packed e2m1 + (N, K // 32) scales -> fp32 (K, N)."""
    table = E2M1.to(w.device)
    kh, N = w.shape
    lo, hi = table[(w & 0xF).long()], table[(w >> 4).long()]
    full = torch.stack([lo, hi], dim=1).view(2 * kh, N)
    return full * torch.exp2(ws.float() - 127).T.repeat_interleave(32, dim=0)


def _check_shape(K: int, N: int):
    if K % BLOCK_K or N % BLOCK_N:
        raise ValueError(f"K={K} and N={N} must be multiples of {BLOCK_K} and {BLOCK_N}")


def grouped_matmul_mx(xq, xs, w, ws, ragged_metadata, out=None):
    """Ragged grouped matmul, one expert per slice: MXFP8 x (rows, K) times MXFP4 w (E, K // 2, N) -> bf16."""
    rows, K = xq.shape
    n_experts, _, N = w.shape
    _check_shape(K, N)
    y = torch.zeros((rows, N), dtype=torch.bfloat16, device=xq.device) if out is None else out
    _grouped_matmul_mx_kernel[(triton.cdiv(N, BLOCK_N) * n_experts,)](
        xq, xs, w, ws, y, ragged_metadata.slice_offs, ragged_metadata.slice_sizes, n_experts, K, N,
        xq.stride(0), xs.stride(0), w.stride(0), w.stride(1), ws.stride(0), ws.stride(1), y.stride(0),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=8, num_stages=2,
    )  # fmt: skip
    return y


def fused_exp_matmul_ep_to_dp_mx(xq, xs, w, ws, expt_assignment, expt_indx_flat, combine_indx, shmem, ragged_metadata):
    """fused_exp_matmul_ep_to_dp with MXFP8 activations and MXFP4 weights (no bias); returns the combine buffer."""
    rows, K = xq.shape
    n_local, _, N = w.shape
    _check_shape(K, N)
    n_slots_per_rank = rows // shmem.get_num_ranks()
    dst_local = shmem.zeros((n_slots_per_rank, N), dtype=torch.bfloat16)
    shmem.barrier()
    bitmask = expt_assignment.expt_bitmask
    _fused_exp_matmul_ep_to_dp_mx_kernel[(triton.cdiv(N, BLOCK_N) * n_local,)](
        dst_local, dst_local.stride(0), xq, xs, w, ws, ragged_metadata.slice_offs, ragged_metadata.slice_sizes,
        bitmask, bitmask.stride(0), expt_indx_flat, combine_indx, n_local, n_slots_per_rank, K, N,
        xq.stride(0), xs.stride(0), w.stride(0), w.stride(1), ws.stride(0), ws.stride(1), shmem.get_heap_bases(),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, SRC_RANK=shmem.get_rank(), N_RANKS=shmem.get_num_ranks(),
        num_warps=8, num_stages=2,
    )  # fmt: skip
    shmem.barrier()
    return dst_local
