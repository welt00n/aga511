import json
import math
import os

import common
from lab import budget, device

MEASUREMENT = os.path.join(common.EP4_FOLDER, "runs", "measure", "2026-11-22_20-25-24-Welton-all", "parts")
LAPTOP = {"sms": 36, "fp32_lanes": 128, "int32_lanes": 128, "fp64_units": 2, "memory": 7.07e9}
A30 = {"sms": 56, "fp32_lanes": 64, "int32_lanes": 64, "fp64_units": 32, "clock_hz": 1.41e9, "memory": 23e9}
FP32_PER_CYCLE_FACTOR = (0.5, 1.0)
FP64_VS_FP32 = (0.6, 0.85)
DISK_BYTES_PER_SECOND = 1.1e9
SERIAL_SETUP_SECONDS = 1.0
CHUNK = 1024


def laptop_speeds():
    with open(os.path.join(MEASUREMENT, "speed.json")) as file:
        rows = json.load(file)["rows"]
    return {(row["precision"], row["sms"]): row for row in rows}


def per_sm_per_cycle(row):
    return row["photons_per_second"]["median"] / row["sms"] / (row["sm_clock_mhz"] * 1e6)


def a30_range(laptop):
    fp32 = per_sm_per_cycle(laptop[(32, 1)])
    low32, high32 = (fp32 * factor * A30["clock_hz"] * A30["sms"] for factor in FP32_PER_CYCLE_FACTOR)
    return {32: (low32, high32), 64: (low32 * FP64_VS_FP32[0], high32 * FP64_VS_FP32[1])}


def duration(seconds):
    return f"{seconds:.0f} s" if seconds < 90 else f"{seconds / 60:.0f} min" if seconds < 5400 else f"{seconds / 3600:.1f} h"


def resident_threads(registers):
    limits = device.KNOWN_TARGETS["80"]
    warps = min(device.warps_allowed_by_registers(registers, limits), device.max_warps(limits))
    return warps * limits["WARP_SIZE"] * A30["sms"]


with common.recorded(__file__):
    laptop = laptop_speeds()
    predicted = a30_range(laptop)
    target = device.target_occupancy("80")
    print("The A30, predicted from the laptop's 1k measurement and the two chips' units per SM (datasheet).")
    print(f"  laptop SM: {LAPTOP['fp32_lanes']} FP32/INT32 lanes, {LAPTOP['fp64_units']} FP64 units; "
          f"A30 SM: {A30['fp32_lanes']} FP32 + {A30['int32_lanes']} INT32 lanes, {A30['fp64_units']} FP64 units, "
          f"~{A30['clock_hz'] / 1e9:.2f} GHz, {A30['sms']} SMs\n")
    for precision in (32, 64):
        one, every = laptop[(precision, 1)], laptop[(precision, 36)]
        low, high = predicted[precision]
        registers = target["precisions"][f"fp{precision}"]["registers"]
        threads = resident_threads(registers)
        minimum = max(low * 1.0, 32 * threads * CHUNK)
        print(f"== FP{precision}")
        print(f"  laptop, measured: one SM {one['photons_per_second']['median']:.3g}/s "
              f"({per_sm_per_cycle(one):.4f} photons per cycle per SM), 36 SMs {every['photons_per_second']['median']:.3g}/s")
        print(f"  A30, predicted:   56 SMs {low:.2g}–{high:.2g}/s = {low / every['photons_per_second']['median']:.1f}–"
              f"{high / every['photons_per_second']['median']:.1f}× the laptop; one SM {low / 56:.2g}–{high / 56:.2g}/s")
        print(f"  A30, compiled:    {registers} registers → occupancy at most "
              f"{target['precisions'][f'fp{precision}']['occupancy_at_most']:.2f}, {threads} resident threads; "
              f"minimum N ≈ {minimum:.1e}")
    print("\nWhat today's runs would take on the A30 (the range comes from the speed range):")
    for label, size, observers, precision, snr in (("the 1k showcase, SNR 10", 1024, 4, 32, 10),
                                                   ("the 2k showcase, SNR 10", 2048, 4, 32, 10),
                                                   ("2k, 7 observers, SNR 10", 2048, 7, 32, 10),
                                                   ("1k at N = 1e14", 1024, 4, 32, None)):
        n = budget.photons(size, 300, budget.nmu_for(300), snr) if snr else 1e14
        passes = budget.passes(size, 300, observers, A30["memory"])
        serial = SERIAL_SETUP_SECONDS + budget.image_bytes(size, 300, observers) / DISK_BYTES_PER_SECOND
        low, high = predicted[precision]
        print(f"  {label}: N = {n:.2g}, {passes} pass(es): {duration(passes * n / high + serial)} – "
              f"{duration(passes * n / low + serial)}")
    print("\nFound on 2026-11-23 (a prediction, to replace with the measurement): FP32 about 0.4–0.8× the laptop,")
    print("FP64 about 4–10× the laptop (full FP64 units, but this kernel is mostly integer and FP32 work).")
