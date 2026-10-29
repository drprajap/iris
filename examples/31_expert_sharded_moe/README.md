# Expert-sharded MoE

An expert-parallel (EP) MoE forward pass on Iris, following the flow of
[`triton_kernels/tests/test_distributed.py`](https://github.com/triton-lang/triton/blob/main/python/triton_kernels/tests/test_distributed.py):
top-k routing, an all-gather of the routing, the DP->EP dispatch, the grouped expert matmul, the
EP->DP combine and the reduce over routes. Each rank owns `n_expts_tot / world_size` experts and
`n_tokens / world_size` tokens.

`fused_exp_matmul_ep_to_dp.py` fuses the expert matmul with the combine: each output tile is stored
straight to the rank that owns its token (`iris.store`), so the combine traffic overlaps the matmul
instead of following it. `mx_expert_matmul.py` provides the same grouped and fused kernels with
MXFP4 weights and MXFP8 activations (`tl.dot_scaled`).

## Usage

Validate the sharded pipeline against a single-device reference:

```terminal
python examples/31_expert_sharded_moe/example_run.py --num_ranks 4 --n_tokens 512 --d_model 256 --n_expts_tot 16 --n_expts_act 4
```

Measure how much of the combine the fused kernel hides (fused, unfused, matmul only, combine only):

```terminal
torchrun --nproc_per_node=4 examples/31_expert_sharded_moe/benchmark.py --tokens 4096
torchrun --nproc_per_node=4 examples/31_expert_sharded_moe/benchmark.py --tokens 4096 --precision mxfp4
```

`hidden = (unfused - fused) / min(matmul only, combine only)`. Each expert is one
`d_model x d_model` matmul; at `d_model=7168` that is about the FLOPs per token of one DeepSeek-V3
expert MLP. `--hot_bias` skews routing toward rank 0's experts.

Run each mode back to back under power sampling, then plot power, GFX clock and temperature:

```terminal
torchrun --nproc_per_node=4 examples/31_expert_sharded_moe/benchmark.py --tokens 4096 --precision mxfp4 \
    --sustain_s 20 --output_dir moe_power
python examples/31_expert_sharded_moe/plot_power.py moe_power/moe_mxfp4_t4096_power
```

Power sampling uses `amdsmi` from the ROCm install (`--amdsmi_python`, `--amdsmi_path`), since the
torch wheels do not ship it. Values are reported as `amdsmi` gives them: socket power in W.

## Results: 4x MI455X (gfx1250), one tray, SPX + NPS1

EP4, 256 experts (64 per rank), top-8, `d_model=7168`, 4096 tokens per rank, random routing.
Medians of 10 samples; ranges cover the four ranks and repeated runs.

| | bf16 | MXFP4 x MXFP8 |
|---|---|---|
| matmul only | 4.13-4.15 ms (~815 TFLOPS) | 2.55-2.56 ms (~1320 TFLOPS) |
| combine only | 0.61-0.67 ms | 0.64-0.69 ms |
| unfused | 4.88-4.92 ms | 3.36-3.38 ms |
| fused | 4.36-4.42 ms | 2.83-2.88 ms |
| combine hidden | 75-84% | 70-86% |

At 1024 tokens per rank the fused kernel hides 24-30% of the combine in both precisions.

Sustained, about 20 s per mode, averaged over the four GPUs (power cap 2.5 kW):

| mode | bf16: ms / power / GFX clock | MXFP4: ms / power / GFX clock |
|---|---|---|
| idle | - / ~1.0 kW / 2.23 GHz | - / ~1.0 kW / 2.23 GHz |
| matmul only | 4.26 / 2.50 kW / 1.68-1.75 GHz | 2.55 / 2.49 kW / 1.93-2.07 GHz |
| combine only | 0.63 / ~1.35 kW / 2.2 GHz | 0.57 / ~1.39 kW / 2.2 GHz |
| unfused | 4.99 / ~2.5 kW / 1.83-1.94 GHz | 3.19 / ~2.22 kW / 1.98-2.08 GHz |
| fused | 4.43 / 2.50 kW / 1.76-1.86 GHz | 2.64 / ~2.36 kW / 1.95-2.09 GHz |

The matmul is what reaches the power cap; the combine alone draws far less. With bf16 every mode
that includes the matmul sits at the cap, and fusing shortens the layer by 11% at the same power.
With MXFP4 the unfused layer stays below the cap, and the fused kernel spends that headroom:
about 140 W more, 17% shorter, about 12% less energy per layer.

These are reference kernels: one 128x128x128 tile configuration, random weights with a constant
scale, and a bf16 dispatch quantized to MXFP8 outside the timed region. The fused, unfused and
combine modes include the wrappers' two host barriers per call.
