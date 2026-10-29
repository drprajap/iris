#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Sample socket power, GFX clock, hotspot temperature and activity on every GPU until SIGTERM.

benchmark.py --sustain_s starts this on rank 0. It needs a python that can import amdsmi, which
ships with ROCm rather than with the torch wheels:

  PYTHONPATH=/opt/rocm/share/amd_smi /usr/bin/python3 power_sampler.py out.csv [interval_s]

Values are written as amdsmi reports them: socket_power in W, power_cap in uW.
"""

import csv
import signal
import sys
import time

import amdsmi

_stop = False


def _on_signal(*_):
    global _stop
    _stop = True


def main():
    out, interval = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    amdsmi.amdsmi_init()
    try:
        handles = amdsmi.amdsmi_get_processor_handles()
        caps = [amdsmi.amdsmi_get_power_cap_info(h)["power_cap"] for h in handles]
        bdfs = [amdsmi.amdsmi_get_gpu_device_bdf(h) for h in handles]
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                ["t", "gpu", "bdf", "socket_power_w", "power_cap_uw", "gfx_clk_mhz", "hotspot_c", "gfx_activity",
                 "umc_activity"]
            )  # fmt: skip
            while not _stop:
                t0 = time.time()
                for i, h in enumerate(handles):
                    power = amdsmi.amdsmi_get_power_info(h)
                    clk = amdsmi.amdsmi_get_clock_info(h, amdsmi.AmdSmiClkType.GFX)["clk"]
                    hot = amdsmi.amdsmi_get_temp_metric(
                        h, amdsmi.AmdSmiTemperatureType.HOTSPOT, amdsmi.AmdSmiTemperatureMetric.CURRENT
                    )
                    act = amdsmi.amdsmi_get_gpu_activity(h)
                    w.writerow(
                        [f"{t0:.4f}", i, bdfs[i], power["socket_power"], caps[i], clk, hot, act["gfx_activity"],
                         act["umc_activity"]]
                    )  # fmt: skip
                f.flush()
                time.sleep(max(0.0, interval - (time.time() - t0)))
    finally:
        amdsmi.amdsmi_shut_down()


if __name__ == "__main__":
    main()
