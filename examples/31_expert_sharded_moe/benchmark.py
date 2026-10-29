#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
How much of the EP->DP combine does the fused expert matmul + combine hide?

Routing, the routing all-gather and the DP->EP dispatch run once. Then, on the same dispatched
tokens, each rank times:

  fused     fused expert matmul + combine (each output tile is stored straight to the token's owner)
  unfused   grouped expert matmul, then convert_ep_to_dp
  gemm      grouped expert matmul alone
  combine   convert_ep_to_dp alone, on a precomputed matmul output
  barrier   the two shmem.barrier() calls the combine and fused wrappers make (overhead reference)

hidden = (unfused - fused) / min(gemm, combine). Each sample starts after a barrier and is host wall
time to torch.cuda.synchronize(). Each expert is one d_model x d_model matmul; at d_model=7168 that
is about the FLOPs per token of one DeepSeek-V3 expert MLP (3 x 7168x2048).

--precision mxfp4 runs mx_expert_matmul.py's MXFP4-weight x MXFP8-activation kernels instead. The
bf16 dispatch output is quantized once, outside the timed modes.

--sustain_s S runs each of --sustain_modes back to back for about S seconds after --idle_s of idle,
while power_sampler.py samples power, GFX clock and temperature on every GPU of the node. Plot the
result with plot_power.py.

  torchrun --nproc_per_node=4 examples/31_expert_sharded_moe/benchmark.py --tokens 4096
  torchrun --nproc_per_node=4 examples/31_expert_sharded_moe/benchmark.py --tokens 4096 --precision mxfp4 \\
      --sustain_s 20 --output_dir moe_power
"""

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time

import torch
import torch.distributed as dist

import iris

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from combine import convert_ep_to_dp  # noqa: E402
from dispatch import convert_dp_to_ep  # noqa: E402
from expert_assignment import make_expt_assignment, make_expt_dict_uniform  # noqa: E402
from fused_exp_matmul_ep_to_dp import fused_exp_matmul_ep_to_dp  # noqa: E402
from grouped_matmul import grouped_matmul  # noqa: E402
from moe import _allgather_iris  # noqa: E402
from mx_expert_matmul import (  # noqa: E402
    fused_exp_matmul_ep_to_dp_mx,
    grouped_matmul_mx,
    quantize_mxfp8,
    random_mxfp4_weights,
)
from ragged_metadata import make_ragged_tensor_metadata, remap_ragged_tensor_metadata  # noqa: E402
from topk import _make_bitmatrix_metadata, topk  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokens", type=int, default=4096, help="tokens per rank")
    p.add_argument("--d_model", type=int, default=7168)
    p.add_argument("--n_expts_tot", type=int, default=256)
    p.add_argument("--n_expts_act", type=int, default=8)
    p.add_argument("--hot_bias", type=float, default=0.0, help="added to rank 0's experts' logits (routing skew)")
    p.add_argument("--precision", choices=["bf16", "mxfp4"], default="bf16")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--heap_size", type=int, default=1 << 35)
    p.add_argument("--output_dir", type=str, default="", help="write per-rank JSON (and power traces) here")
    p.add_argument("--sustain_s", type=float, default=0.0, help="if >0: run each sustain mode this long")
    p.add_argument("--sustain_modes", type=str, default="gemm,combine,unfused,fused")
    p.add_argument("--idle_s", type=float, default=3.0, help="idle time before each sustained mode")
    p.add_argument("--sample_ms", type=float, default=50.0, help="power sampling interval")
    p.add_argument("--amdsmi_python", type=str, default="/usr/bin/python3", help="python that can import amdsmi")
    p.add_argument("--amdsmi_path", type=str, default="/opt/rocm/share/amd_smi", help="PYTHONPATH for amdsmi")
    return p.parse_args()


def reuse_heap_buffers(ctx):
    """The example's wrappers allocate a fresh heap buffer on every call and the heap does not free.
    Return one zeroed buffer per (shape, dtype) instead, so repeated calls fit in the heap with any
    allocator."""
    alloc, cache = ctx.zeros, {}

    def zeros(*args, **kwargs):
        key = (repr(args), kwargs.get("dtype"))
        if key not in cache:
            cache[key] = alloc(*args, **kwargs)
        else:
            cache[key].zero_()
        return cache[key]

    ctx.zeros = zeros


def sustain(args, ctx, rank, modes, res):
    """Run each sustain mode back to back for about args.sustain_s seconds while rank 0 samples power."""
    base = os.path.join(args.output_dir, f"moe_{args.precision}_t{args.tokens}_power")
    sampler = None
    if rank == 0:
        env = dict(os.environ, PYTHONPATH=args.amdsmi_path)
        sampler = subprocess.Popen(
            [args.amdsmi_python, os.path.join(HERE, "power_sampler.py"), f"{base}.csv", str(args.sample_ms / 1e3)],
            env=env,
        )
    phases = []
    try:
        for name in args.sustain_modes.split(","):
            fn = modes[name]
            # Every rank must make the same number of calls: the wrappers' host barriers pair up across ranks.
            n_t = torch.tensor([max(1, math.ceil(args.sustain_s * 1e3 / res[name]))], dtype=torch.int64)
            dist.broadcast(n_t, src=0)
            n = int(n_t.item())
            ctx.barrier()
            t_idle = time.time()
            time.sleep(args.idle_s)
            ctx.barrier()
            t0 = time.time()
            for _ in range(n):
                fn()
            torch.cuda.synchronize()
            ctx.barrier()
            t1 = time.time()
            phases.append(
                {
                    "mode": name,
                    "idle_start": t_idle,
                    "start": t0,
                    "end": t1,
                    "iters": n,
                    "ms_per_iter": (t1 - t0) * 1e3 / n,
                }
            )
            if rank == 0:
                print(f"[sustain] {name}: {n} iters in {t1 - t0:.1f} s ({(t1 - t0) * 1e3 / n:.3f} ms/iter)", flush=True)
        ctx.barrier()
        time.sleep(args.idle_s)
    finally:
        if sampler is not None:
            sampler.terminate()
            sampler.wait(timeout=10)
    if rank == 0:
        with open(f"{base}_phases.json", "w") as f:
            json.dump({"args": vars(args), "bench_ms": res, "phases": phases}, f)
        print(f"power trace: {base}.csv  (plot: python {os.path.join(HERE, 'plot_power.py')} {base})", flush=True)


def main():
    args = parse_args()
    if args.sustain_s > 0 and not args.output_dir:
        raise SystemExit("--sustain_s needs --output_dir for the power trace")
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    ctx = iris.iris(heap_size=args.heap_size)
    reuse_heap_buffers(ctx)
    rank, world_size = ctx.get_rank(), ctx.get_num_ranks()
    device = torch.device("cuda", local_rank)
    T, D, E, K = args.tokens, args.d_model, args.n_expts_tot, args.n_expts_act

    g = torch.Generator(device=device).manual_seed(1000 + rank)
    x_dp = torch.randn(T, D, device=device, dtype=torch.bfloat16, generator=g)
    logits = torch.rand(T, E, device=device, dtype=torch.float32, generator=g)
    expt_dict = make_expt_dict_uniform(world_size, E)
    assign = make_expt_assignment(world_size, E, expt_dict, device)
    logits[:, assign.expt_boolmask[0]] += args.hot_bias
    gw = torch.Generator(device=device).manual_seed(2000 + rank)
    n_local = int(assign.expt_boolmask[rank].sum())

    # Routing + dispatch, once (the same steps as moe.mixture_of_expt_epsharded).
    local_topk = topk(logits, K, apply_softmax=True)
    indx_global = _allgather_iris(local_topk.indx.contiguous().to(torch.int32), ctx)
    meta = _make_bitmatrix_metadata(indx_global.to(torch.int32), E)
    n_active = int(meta.col_sum.sum().item())
    x_meta = make_ragged_tensor_metadata(meta.col_sum, n_active)
    y_ep = convert_dp_to_ep(x_dp, assign, indx_global, meta.row_sorted_indx, ctx).clone()
    expt_map = assign.expt_map[rank, :].contiguous()
    y_ep_meta = remap_ragged_tensor_metadata(x_meta, expt_map)
    flat_expt = indx_global.to(torch.int32).reshape(-1)
    combine_indx = meta.col_sorted_indx
    rows_here = int(meta.col_sum[assign.expt_boolmask[rank]].sum())

    if args.precision == "mxfp4":
        w_q, w_s = random_mxfp4_weights(n_local, D, D, device, gw)
        y_q, y_s = quantize_mxfp8(y_ep)

        def gemm():
            return grouped_matmul_mx(y_q, y_s, w_q, w_s, y_ep_meta)

        def fused():
            return fused_exp_matmul_ep_to_dp_mx(y_q, y_s, w_q, w_s, assign, flat_expt, combine_indx, ctx, y_ep_meta)
    else:
        w_ep = (torch.randn(n_local, D, D, device=device, dtype=torch.bfloat16, generator=gw) * D**-0.5).contiguous()
        b_ep = torch.zeros(n_local, D, device=device, dtype=torch.float32)

        def gemm():
            return grouped_matmul(y_ep, w_ep, b_ep, y_ep_meta)

        def fused():
            return fused_exp_matmul_ep_to_dp(
                y_ep, w_ep, b_ep, assign, expt_map, flat_expt, combine_indx, ctx, ragged_metadata=y_ep_meta
            )

    gemm_out = gemm()
    modes = {
        "fused": fused,
        "unfused": lambda: convert_ep_to_dp(gemm(), assign, flat_expt, combine_indx, ctx),
        "gemm": gemm,
        "combine": lambda: convert_ep_to_dp(gemm_out, assign, flat_expt, combine_indx, ctx),
        "barrier": lambda: (ctx.barrier(), ctx.barrier()),
    }

    fused_out = modes["fused"]().clone()
    ctx.barrier()
    unfused_out = modes["unfused"]().clone()
    ctx.barrier()
    max_err = (fused_out.float() - unfused_out.float()).abs().max().item()
    valid = bool(torch.allclose(fused_out.float(), unfused_out.float(), atol=2e-2, rtol=2e-2))

    def sample(fn):
        ctx.barrier()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1e3

    res = {}
    for name, fn in modes.items():
        for _ in range(2):
            sample(fn)
        torch.cuda.nvtx.range_push(f"mode:{name}")
        res[name] = statistics.median(sample(fn) for _ in range(args.iters))
        torch.cuda.nvtx.range_pop()

    saved = res["unfused"] - res["fused"]
    hideable = min(res["gemm"], res["combine"])
    tflops = 2 * rows_here * D * D / (res["gemm"] * 1e-3) / 1e12
    print(
        f"[rank {rank}] {args.precision} rows={rows_here} valid={valid} max|fused-unfused|={max_err:.3g} "
        + " ".join(f"{k}={v:.3f}ms" for k, v in res.items())
        + f"  gemm {tflops:.0f} TFLOPS  fused hides {saved:.3f} of {hideable:.3f} ms ({100 * saved / hideable:.0f}%)",
        flush=True,
    )
    if args.output_dir:
        out = os.path.join(args.output_dir, f"moe_{args.precision}_t{args.tokens}_rank{rank}.json")
        with open(out, "w") as f:
            json.dump(
                {"rank": rank, "world_size": world_size, "args": vars(args), "rows_here": rows_here, "valid": valid,
                 "max_err": max_err, "ms": res, "gemm_tflops": tflops, "hidden_pct": 100 * saved / hideable},
                f,
            )  # fmt: skip
    if args.sustain_s > 0:
        sustain(args, ctx, rank, modes, res)
    ctx.barrier()
    dist.destroy_process_group()
    if not valid:
        sys.exit(1)


if __name__ == "__main__":
    main()
