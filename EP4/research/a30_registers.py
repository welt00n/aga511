import os

import common

TARGETS = {"this GPU": ("native", None), "A30": ("80", common.A30_LIMITS)}


def compiled_resources(precision, architecture, folder):
    flags = ["-arch=native"] if architecture == "native" else [f"-arch=sm_{architecture}"]
    report = common.compile_kernel(precision, os.path.join(folder, f"gpu{precision}_{architecture}.cubin"),
                                   [*flags, "-cubin", "-Xptxas", "-v"])
    return common.kernel_resources(report)


def describe(name, architecture, limits, folder):
    limits = limits or common.sm_limits()
    for precision in (64, 32):
        registers, spills = compiled_resources(precision, architecture, folder)
        warps = min(common.warps_allowed_by_registers(registers, limits), common.max_warps(limits))
        print(f"{name:9s} {limits['NAME'][:36]:36s} FP{precision}: {registers:3d} registers, {spills} B spilled -> "
              f"{common.registers_per_warp(registers, limits)} per warp, {warps} of {common.max_warps(limits)} warps "
              f"({warps * limits['WARP_SIZE']} threads per SM), occupancy at most "
              f"{common.occupancy_allowed_by_registers(registers, limits):.2f}")


with common.recorded(__file__):
    print("How many registers will the kernel use on the A30? Compile only (ptxas -v), no A30 needed.\n")
    folder = common.scratch_folder("a30_registers")
    for name, (architecture, limits) in TARGETS.items():
        describe(name, architecture, limits, folder)
    print("\nFound on 2026-11-22: sm_80 uses 96 (FP64) and 76 (FP32) registers, no spills, against 80 / 67 on")
    print("sm_120: occupancy at most 0.33 / 0.39 on the A30 (672 / 800 threads per SM). Why: front_end_registers.py.")
