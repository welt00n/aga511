import json
import os
import shutil
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import common  # noqa: E402
from lab import engine, scaling  # noqa: E402
from lab.gpu_monitor import GpuMonitor  # noqa: E402

PHOTONS = {64: 10 ** 9, 32: 10 ** 10}
CAPS = {64: [None, 72, 64, 56, 48, 40, 32], 32: [None, 64, 56, 48, 40, 32]}
REPEATS = 3
COLORS = {64: scaling.PHOTON_PHASE, 32: scaling.WHOLE_RUN}
FOLDER = os.path.join(common.RESULTS_FOLDER, "register_sweep")


def variant(cap):
    if cap is None:
        return None
    engine.GPU_VARIANTS[f"registers_{cap}"] = [f"-maxrregcount={cap}"]
    return f"registers_{cap}"


def run_once(precision, cap):
    config = engine.default_star_config(PHOTONS[precision], variant=variant(cap))
    folder = common.scratch_folder("register_sweep")
    with GpuMonitor() as monitor:
        output = engine.engine_output(engine.engine_command(config, folder, "gpu", precision, None))
    image = engine.load_and_delete_observers(folder)[1]
    measured = engine.lines_of_kind(output, "SUMMARY")[0]
    measured.update(engine.lines_of_kind(output, "KERNEL")[0])
    measured["CYCLES"] = engine.lines_of_kind(output, "SMS")[0]["CYCLES"]
    measured.update(monitor.summary())
    return measured, image


def compiled_spills(precision, cap):
    report = engine.read_text(engine.binary_path("gpu", precision, variant(cap)) + ".report") or ""
    return common.kernel_resources(report)[1]


def interleaved_runs(precision):
    measured = {cap: [] for cap in CAPS[precision]}
    images = {}
    for repeat in range(REPEATS):
        order = CAPS[precision] if repeat % 2 == 0 else CAPS[precision][::-1]
        for cap in order:
            summary, images[cap] = run_once(precision, cap)
            measured[cap].append(summary)
            print(f"  FP{precision} repeat {repeat + 1} cap {cap or 'none'}: {summary['REGISTERS_PER_THREAD']} registers, "
                  f"kernel {summary['KERNEL_TIME']:.3f} s, {summary['CYCLES']:.3e} cycles", flush=True)
    return measured, images


def row(precision, cap, measured, image, baseline_image, limits):
    first = measured[0]
    times = [run["KERNEL_TIME"] for run in measured]
    cycles = [run["CYCLES"] for run in measured]
    registers = first["REGISTERS_PER_THREAD"]
    return {"precision": precision, "cap": cap or "none", "registers": registers,
            "registers_per_warp": common.registers_per_warp(registers, limits),
            "warps_allowed_by_registers": common.warps_allowed_by_registers(registers, limits),
            "threads_per_block": first["THREADS_PER_BLOCK"], "blocks_per_sm": first["BLOCKS_PER_UNIT"],
            "occupancy": first["OCCUPANCY"], "spill_bytes": compiled_spills(precision, cap),
            "kernel_time_median": statistics.median(times), "cycles_median": statistics.median(cycles),
            "effective_mhz": statistics.median(c / t / 1e6 for c, t in zip(cycles, times)),
            "power_median_w": statistics.median(run.get("POWER_W_MEDIAN") or 0 for run in measured),
            "photons_per_second": PHOTONS[precision] / statistics.median(times),
            "image_equals_uncapped": bool(np.array_equal(image, baseline_image))}


def sweep(precision, limits):
    measured, images = interleaved_runs(precision)
    return [row(precision, cap, measured[cap], images[cap], images[None], limits) for cap in CAPS[precision]]


def table(rows):
    base = rows[0]
    lines = [f"FP{base['precision']}, N = {PHOTONS[base['precision']]:.0e}, {REPEATS} interleaved repeats",
             f"{'cap':>5s} {'regs':>5s} {'warps':>6s} {'occ':>5s} {'spill B':>8s} {'kernel s':>9s} {'time':>6s} "
             f"{'cycles':>7s} {'MHz':>6s} {'W':>5s} {'same image':>11s}"]
    for item in rows:
        lines.append(f"{item['cap']:>5} {item['registers']:5d} {item['warps_allowed_by_registers']:6d} "
                     f"{item['occupancy']:5.2f} {item['spill_bytes']:8d} {item['kernel_time_median']:9.3f} "
                     f"{item['kernel_time_median'] / base['kernel_time_median']:6.3f} "
                     f"{item['cycles_median'] / base['cycles_median']:7.3f} {item['effective_mhz']:6.0f} "
                     f"{item['power_median_w']:5.0f} {'yes' if item['image_equals_uncapped'] else 'NO':>11s}")
    return lines


def figure(results, limits, path):
    fig, axes = plt.subplots(2, 1, figsize=(8, 7.5), sharex=True)
    registers = np.arange(24, 97)
    axes[0].step(registers, [common.occupancy_allowed_by_registers(r, limits) for r in registers], where="post",
                 color=scaling.IDEAL, linewidth=1.2, label="what the registers allow")
    for precision, rows in results.items():
        x = [item["registers"] for item in rows]
        axes[0].plot(x, [item["occupancy"] for item in rows], "o", color=COLORS[precision], markersize=8,
                     label=f"FP{precision}: the launch")
        relative = [rows[0]["cycles_median"] / item["cycles_median"] for item in rows]
        axes[1].plot(x, relative, "o-", color=COLORS[precision], linewidth=2, markersize=7, label=f"FP{precision}")
    axes[1].axhline(1, color=scaling.IDEAL, linewidth=1, linestyle=":")
    scaling.styled_axes(axes[0], "Occupancy: the registers per thread decide how many warps fit", "",
                        f"occupancy (warps resident / {common.max_warps(limits)})")
    scaling.styled_axes(axes[1], "Speed in cycles (clock-independent): uncapped cycles / capped cycles",
                        "registers per thread (after the cap)", "speed vs uncapped")
    for ax in axes:
        ax.legend(frameon=False, fontsize=8, labelcolor=scaling.TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


with common.recorded(__file__):
    limits = common.sm_limits()
    print("Cap the registers (-maxrregcount) and measure occupancy and speed. The caps are interleaved (forward,")
    print("then backward) so a clock that drifts during the sweep hits every cap alike; speed is also given in")
    print("cycles of the slowest SM, which do not depend on the clock.\n")
    results = {precision: sweep(precision, limits) for precision in CAPS}
    print()
    for rows in results.values():
        print("\n".join(table(rows)) + "\n")
    shutil.rmtree(FOLDER, ignore_errors=True)
    os.makedirs(FOLDER)
    with open(os.path.join(FOLDER, "register_sweep.json"), "w") as file:
        json.dump({"limits": limits, "photons": PHOTONS, "repeats": REPEATS, "results": results}, file, indent=1)
    figure(results, limits, os.path.join(FOLDER, "registers_and_occupancy.png"))
    print("Found on 2026-11-22 (interleaved, clock steady): FP64 occupancy follows the register limit; up to 0.67")
    print("(64 registers) the speed is the same, then spills cost up to ~10% in cycles. FP32 is fastest uncapped:")
    print("55 registers (0.75, no spills) is 1% slower, 32 (1.00) 5%. Images identical at every cap.")
