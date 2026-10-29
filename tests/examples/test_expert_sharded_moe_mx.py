#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

import iris


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PROJECT_ROOT = Path(__file__).resolve()
while not (PROJECT_ROOT / "tests").is_dir() or not (PROJECT_ROOT / "examples").is_dir():
    if PROJECT_ROOT == PROJECT_ROOT.parent:
        raise FileNotFoundError("Could not find project root")
    PROJECT_ROOT = PROJECT_ROOT.parent

EXAMPLE_DIR = PROJECT_ROOT / "examples" / "31_expert_sharded_moe"
sys.path.insert(0, str(EXAMPLE_DIR))
MX = _load_module("mx_expert_matmul_31_moe", EXAMPLE_DIR / "mx_expert_matmul.py")
RAGGED = _load_module("ragged_metadata_31_moe", EXAMPLE_DIR / "ragged_metadata.py")
EXPERT_ASSIGNMENT = _load_module("expert_assignment_31_moe", EXAMPLE_DIR / "expert_assignment.py")
TOPK = _load_module("topk_31_moe", EXAMPLE_DIR / "topk.py")
DISPATCH = _load_module("dispatch_31_moe", EXAMPLE_DIR / "dispatch.py")
COMBINE = _load_module("combine_31_moe", EXAMPLE_DIR / "combine.py")
MOE = _load_module("moe_31_moe", EXAMPLE_DIR / "moe.py")


@pytest.mark.parametrize("K, N, sizes", [(512, 256, [130, 0, 64, 300]), (1024, 384, [200, 129])])
def test_grouped_matmul_mx_matches_dequantized_reference(K, N, sizes):
    device = torch.device("cuda", torch.cuda.current_device())
    g = torch.Generator(device=device).manual_seed(0)
    rows = sum(sizes)
    x = torch.randn(rows, K, device=device, dtype=torch.bfloat16, generator=g)
    xq, xs = MX.quantize_mxfp8(x)
    w, ws = MX.random_mxfp4_weights(len(sizes), K, N, device, g)
    meta = RAGGED.make_ragged_tensor_metadata(torch.tensor(sizes, dtype=torch.int32, device=device), rows)

    y = MX.grouped_matmul_mx(xq, xs, w, ws, meta)

    xd = MX.dequant_mxfp8(xq, xs)
    off = 0
    for e, n in enumerate(sizes):
        if n:
            ref = xd[off : off + n] @ MX.dequant_mxfp4(w[e], ws[e])
            rel = ((y[off : off + n].float() - ref).abs().max() / ref.abs().max()).item()
            assert rel < 1e-2, f"expert {e}: max relative error {rel:.3e}"
        off += n


@pytest.mark.parametrize("n_tokens, d_model, n_expts_act", [(256, 256, 2)])
def test_fused_mx_matches_unfused_mx(n_tokens, d_model, n_expts_act):
    if not dist.is_initialized():
        pytest.skip("torch.distributed not initialized")

    shmem = None
    try:
        shmem = iris.iris(1 << 30)
        rank = shmem.get_rank()
        world_size = shmem.get_num_ranks()
        device = torch.device(f"cuda:{rank}")
        n_expts_tot = world_size * 2
        if n_tokens % world_size != 0:
            pytest.skip("n_tokens must be divisible by world size for this test")
        n_tokens_local = n_tokens // world_size

        torch.manual_seed(0)
        x_global = torch.randn(n_tokens, d_model, device=device, dtype=torch.bfloat16)
        l_global = torch.rand(n_tokens, n_expts_tot, device=device, dtype=torch.float32)
        dist.broadcast(x_global, src=0)
        dist.broadcast(l_global, src=0)
        expt_dict = EXPERT_ASSIGNMENT.make_expt_dict_uniform(world_size, n_expts_tot)
        assign = EXPERT_ASSIGNMENT.make_expt_assignment(world_size, n_expts_tot, expt_dict, device)
        first = rank * n_tokens_local
        x_dp = x_global[first : first + n_tokens_local].contiguous()
        l_dp = l_global[first : first + n_tokens_local].contiguous()
        n_local = int(assign.expt_boolmask[rank].sum())
        w, ws = MX.random_mxfp4_weights(
            n_local, d_model, d_model, device, torch.Generator(device=device).manual_seed(10 + rank)
        )

        local_topk = TOPK.topk(l_dp, n_expts_act, apply_softmax=True)
        indx_global = MOE._allgather_iris(local_topk.indx.contiguous().to(torch.int32), shmem)
        meta = TOPK._make_bitmatrix_metadata(indx_global.to(torch.int32), n_expts_tot)
        x_meta = RAGGED.make_ragged_tensor_metadata(meta.col_sum, int(meta.col_sum.sum().item()))
        y_ep = DISPATCH.convert_dp_to_ep(x_dp, assign, indx_global, meta.row_sorted_indx, shmem)
        y_meta = RAGGED.remap_ragged_tensor_metadata(x_meta, assign.expt_map[rank, :].contiguous())
        flat_expt = indx_global.to(torch.int32).reshape(-1)
        yq, ys = MX.quantize_mxfp8(y_ep)

        fused = MX.fused_exp_matmul_ep_to_dp_mx(yq, ys, w, ws, assign, flat_expt, meta.col_sorted_indx, shmem, y_meta)
        fused = fused.clone()
        unfused = COMBINE.convert_ep_to_dp(
            MX.grouped_matmul_mx(yq, ys, w, ws, y_meta), assign, flat_expt, meta.col_sorted_indx, shmem
        )
        torch.cuda.synchronize()
        assert torch.equal(fused, unfused), f"rank {rank}: fused and unfused MX outputs differ"
    finally:
        if shmem is not None:
            shmem.barrier()
            del shmem
