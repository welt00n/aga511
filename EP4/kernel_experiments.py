import argparse
import json
import os
import re
import shutil
import statistics
from datetime import datetime

import numpy as np

import engine
from gpu_monitor import GpuMonitor

RUNS_FOLDER = os.path.join(engine.HERE, "runs")
DEFAULT_PHOTONS = {64: 5 * 10 ** 8, 32: 5 * 10 ** 9}
VARIANTS = [None, "scene_in_global", "registers_64", "registers_40", "no_fma", "sections"]
SPILLS = re.compile(r"(\d+) bytes spill stores")


def spill_bytes(precision, variant):
    report = engine.read_text(engine.binary_path("gpu", precision, variant) + ".report") or ""
    kernel_part = report[report.find("trace_photons"):]
    match = SPILLS.search(kernel_part)
    return int(match.group(1)) if match else None


def run_once(config, precision, folder):
    with GpuMonitor() as monitor:
        output = engine.run_engine_output(config, folder, "gpu", precision)
    summary = engine.lines_of_kind(output, "SUMMARY")[0]
    summary.update(engine.lines_of_kind(output, "KERNEL")[0])
    summary.update(monitor.summary())
    summary["SECTIONS"] = engine.lines_of_kind(output, "SECTION")
    return summary


def run_variant(precision, variant, photons, repeats, folder):
    config = engine.default_star_config(photons, variant=variant)
    runs = []
    for repeat in range(repeats):
        output_dir = os.path.join(folder, f"images-{variant or 'baseline'}-{precision}")
        shutil.rmtree(output_dir, ignore_errors=True)
        runs.append(run_once(config, precision, output_dir))
    image = engine.load_observers(output_dir)[1]
    shutil.rmtree(output_dir)
    return runs, image


def describe_variant(name, runs, image, baseline_image, precision, variant):
    kernel_times = [run["KERNEL_TIME"] for run in runs]
    first = runs[0]
    differing = int(np.count_nonzero(image != baseline_image)) if baseline_image is not None else 0
    return {"variant": name, "registers": first["REGISTERS_PER_THREAD"], "spill_bytes": spill_bytes(precision, variant),
            "threads_per_block": first["THREADS_PER_BLOCK"], "occupancy": first["OCCUPANCY"],
            "kernel_time_median": statistics.median(kernel_times), "kernel_time_min": min(kernel_times),
            "kernel_time_max": max(kernel_times),
            "photons_per_second": first["PHOTONS"] / statistics.median(kernel_times),
            "sm_clock_median_mhz": statistics.median(run.get("SM_CLOCK_MHZ_MEDIAN", 0) for run in runs),
            "power_median_w": statistics.median(run.get("POWER_W_MEDIAN", 0) for run in runs),
            "pixels_differing_from_baseline": differing, "sections": first["SECTIONS"]}


def table_lines(precision, rows):
    baseline = rows[0]["kernel_time_median"]
    lines = [f"## FP{precision}, N = {DEFAULT_PHOTONS[precision]:.0e}", "",
             "| variant | registers | spills (B) | threads/block | occupancy | kernel time (median, min–max) | "
             "photons/s | vs baseline | SM clock (MHz) | power (W) | pixels ≠ baseline |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['variant']} | {row['registers']} | {row['spill_bytes']} | {row['threads_per_block']} | "
                     f"{row['occupancy']:.3f} | {row['kernel_time_median']:.4f} s "
                     f"({row['kernel_time_min']:.4f}–{row['kernel_time_max']:.4f}) | "
                     f"{row['photons_per_second']:.3g} | {baseline / row['kernel_time_median']:.3f}× | "
                     f"{row['sm_clock_median_mhz']:.0f} | {row['power_median_w']:.1f} | "
                     f"{row['pixels_differing_from_baseline']} |")
    sections = next((row["sections"] for row in rows if row["sections"]), [])
    if sections:
        lines += ["", "**Where the cycles go (the `sections` variant):** " +
                  ", ".join(f"{section['NAME']} {100 * section['SHARE']:.1f}%" for section in sections)]
    return lines


def main():
    parser = argparse.ArgumentParser(description="The GPU kernel's variants: memory, registers, FMA, section timers")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--precisions", type=int, nargs="*", default=[64, 32])
    args = parser.parse_args()
    folder = os.path.join(RUNS_FOLDER, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "-kernel-experiments")
    os.makedirs(folder)
    lines, data = ["# Kernel experiments (the laptop GPU; the whole GPU, one pass, 4 observers, 200², 150 frames)"], {}
    for precision in args.precisions:
        rows, baseline_image = [], None
        for variant in VARIANTS:
            runs, image = run_variant(precision, variant, DEFAULT_PHOTONS[precision], args.repeats, folder)
            rows.append(describe_variant(variant or "baseline", runs, image, baseline_image, precision, variant))
            baseline_image = image if variant is None else baseline_image
            print(f"FP{precision} {variant or 'baseline'}: {rows[-1]['kernel_time_median']:.4f} s", flush=True)
        lines += [""] + table_lines(precision, rows)
        data[precision] = rows
    with open(os.path.join(folder, "report.md"), "w") as file:
        file.write("\n".join(lines) + "\n")
    with open(os.path.join(folder, "data.json"), "w") as file:
        json.dump(data, file, indent=2)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
