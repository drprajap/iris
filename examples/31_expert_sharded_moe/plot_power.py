#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Plot a benchmark.py --sustain_s power trace: socket power, GFX clock and hotspot per GPU over time,
with each sustained mode shaded, and print a per-mode summary over the middle 80% of its window.

  python plot_power.py <output_dir>/moe_<precision>_t<tokens>_power
"""

import csv
import json
import statistics
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {"gemm": "#2b6cb0", "combine": "#dd6b20", "unfused": "#805ad5", "fused": "#38a169"}
FIELDS = ("socket_power_w", "gfx_clk_mhz", "hotspot_c", "gfx_activity")


def load(base):
    meta = json.load(open(f"{base}_phases.json"))
    rows = list(csv.DictReader(open(f"{base}.csv")))
    t_origin = float(rows[0]["t"])
    series = defaultdict(lambda: defaultdict(list))
    for r in rows:
        g = int(r["gpu"])
        series[g]["t"].append(float(r["t"]) - t_origin)
        for k in FIELDS:
            series[g][k].append(float(r[k]) if r[k] not in ("N/A", "") else float("nan"))
    return meta, series, t_origin, float(rows[0]["power_cap_uw"]) / 1e6


def main(base):
    meta, series, t_origin, cap_w = load(base)
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True, constrained_layout=True)
    for ax, key, label in zip(axes, FIELDS, ("socket power (W)", "GFX clock (MHz)", "hotspot (C)")):
        for g in sorted(series):
            ax.plot(series[g]["t"], series[g][key], lw=0.9, label=f"GPU {g}")
        for ph in meta["phases"]:
            ax.axvspan(ph["start"] - t_origin, ph["end"] - t_origin, color=COLORS.get(ph["mode"], "#999"), alpha=0.15)
        ax.set_ylabel(label)
    axes[0].axhline(cap_w, color="red", ls="--", lw=0.8, label=f"power cap {cap_w:.0f} W")
    for ph in meta["phases"]:
        axes[0].text(
            (ph["start"] + ph["end"]) / 2 - t_origin,
            axes[0].get_ylim()[1],
            ph["mode"],
            ha="center",
            va="top",
            fontsize=9,
        )
    axes[0].legend(loc="upper right", fontsize=8, ncol=5)
    axes[-1].set_xlabel("time (s)")
    a = meta["args"]
    fig.suptitle(
        f"Iris example 31 expert matmul + combine, sustained ({a['precision']}): tokens/rank={a['tokens']} "
        f"d_model={a['d_model']} experts={a['n_expts_tot']} top-{a['n_expts_act']}  (shaded: mode running)",
        fontsize=11,
    )
    fig.savefig(f"{base}.png", dpi=100)
    print(f"{base}.png")

    print("mode,ms_per_iter,gpu,power_w_mean,power_w_max,gfx_clk_mhz_mean,hotspot_c_max,gfx_activity_mean")
    first = meta["phases"][0]
    windows = [("idle", first["idle_start"], first["start"], None)]
    windows += [(ph["mode"], ph["start"], ph["end"], ph["ms_per_iter"]) for ph in meta["phases"]]
    for mode, s, e, ms in windows:
        lo, hi = s + 0.1 * (e - s) - t_origin, e - 0.1 * (e - s) - t_origin
        for g in sorted(series):
            idx = [i for i, t in enumerate(series[g]["t"]) if lo <= t <= hi]
            pw = [series[g]["socket_power_w"][i] for i in idx]
            ck = [series[g]["gfx_clk_mhz"][i] for i in idx]
            hs = [series[g]["hotspot_c"][i] for i in idx]
            ac = [series[g]["gfx_activity"][i] for i in idx]
            ms_s = f"{ms:.3f}" if ms is not None else ""
            print(
                f"{mode},{ms_s},{g},{statistics.mean(pw):.0f},{max(pw):.0f},{statistics.mean(ck):.0f},{max(hs):.0f},"
                f"{statistics.mean(ac):.0f}"
            )


if __name__ == "__main__":
    main(sys.argv[1])
