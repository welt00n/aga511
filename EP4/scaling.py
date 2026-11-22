import argparse
import glob
import json
import os
import statistics
from collections import defaultdict
from datetime import datetime

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import engine

RUNS_FOLDER = os.path.join(engine.HERE, "runs")
SWEEP_LABELS = ["strong-gpu64", "strong-gpu32", "strong-gpu32-n1e+08", "weak-gpu64", "weak-gpu32",
                "problem-size-gpu32", "devices"]
ALL_SMS = 36

PHOTON_PHASE = "#2a78d6"
WHOLE_RUN = "#eb6834"
CYCLES = "#1baf7a"
IDEAL = "#8a8985"
TEXT = "#52514e"
GRID = "#e4e3df"


# ============================================================ reading the sweeps

def latest_sweep_folder(label):
    folders = sorted(glob.glob(os.path.join(RUNS_FOLDER, f"*-sweep-{label}")))
    return folders[-1] if folders else None


def read_points(folder):
    points = []
    for path in sorted(glob.glob(os.path.join(folder, "point_*.json"))):
        with open(path) as file:
            points.append(json.load(file))
    return points


def photon_time(point):
    return point["SUMMARY"]["PHOTONS_TIME"]


def whole_run_time(point):
    return point["SUMMARY"]["TIME"]


def kernel_cycles(point):
    return point["SMS"][0]["CYCLES"] if point["SMS"] else None


def sm_clock(point):
    return point["GPU_MONITOR"].get("SM_CLOCK_MHZ_MEDIAN")


def serial_time(point):
    return whole_run_time(point) - photon_time(point)


def medians_by(points, key, measure):
    groups = defaultdict(list)
    for point in points:
        value = measure(point)
        if value is not None:
            groups[key(point)].append(value)
    return {group: statistics.median(values) for group, values in sorted(groups.items())}


def by_sms(point):
    return point["sms"]


# ============================================================ the laws

def fit_amdahl(sms, times):
    n = np.array(sms, dtype=float)
    columns = np.column_stack([np.ones_like(n), 1 / n])
    (serial, parallel), *_ = np.linalg.lstsq(columns, np.array(times), rcond=None)
    one_unit = serial + parallel
    return {"s": serial / one_unit, "p": parallel / one_unit, "t1": one_unit, "ceiling": one_unit / serial}


def amdahl_speedup(fit, n):
    return 1 / (fit["s"] + fit["p"] / np.asarray(n, dtype=float))


def fit_gustafson(sms, scaled_speedups):
    n = np.array(sms, dtype=float)
    shortfall = n - np.array(scaled_speedups)
    return {"s_prime": float(np.sum(shortfall * (n - 1)) / np.sum((n - 1) ** 2))}


def parallel_fraction(speedup, units=ALL_SMS):
    return (1 - 1 / speedup) / (1 - 1 / units)


# ============================================================ figures

def styled_axes(ax, title, xlabel, ylabel):
    ax.set_title(title, color=TEXT, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=TEXT)
    ax.set_ylabel(ylabel, color=TEXT)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def save_figure(fig, folder, name, data):
    fig.tight_layout()
    fig.savefig(os.path.join(folder, name + ".png"), dpi=150)
    plt.close(fig)
    with open(os.path.join(folder, name + ".json"), "w") as file:
        json.dump(data, file, indent=2)


def plot_series(ax, x, y, color, label, fit_x=None, fit_y=None):
    ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=6, label=label)
    if fit_x is not None:
        ax.plot(fit_x, fit_y, color=color, linewidth=1.5, linestyle="--", alpha=0.8)


# ============================================================ strong scaling (Amdahl)

def analyse_strong(points, precision, folder, name):
    photons = medians_by(points, by_sms, photon_time)
    totals = medians_by(points, by_sms, whole_run_time)
    cycles = medians_by(points, by_sms, kernel_cycles)
    clocks = medians_by(points, by_sms, sm_clock)
    serial = medians_by(points, by_sms, serial_time)
    sms = sorted(photons)
    if len(sms) < 3 or 1 not in sms:
        return None
    photon_fit = fit_amdahl(sms, [photons[n] for n in sms])
    total_fit = fit_amdahl(sms, [totals[n] for n in sms])
    speedup = {n: photons[1] / photons[n] for n in sms}
    total_speedup = {n: totals[1] / totals[n] for n in sms}
    cycle_speedup = {n: cycles[1] / cycles[n] for n in sms if n in cycles and 1 in cycles}
    measured_s = statistics.median(serial.values()) / totals[1]

    fig, (ax_speedup, ax_clock) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [2, 1]})
    curve = np.linspace(1, max(sms), 200)
    ax_speedup.plot(curve, curve, color=IDEAL, linewidth=1, linestyle=":", label="ideal: S = n")
    plot_series(ax_speedup, sms, [speedup[n] for n in sms], PHOTON_PHASE,
                f"photon phase (Amdahl fit p = {photon_fit['p']:.4f})", curve, amdahl_speedup(photon_fit, curve))
    plot_series(ax_speedup, sms, [total_speedup[n] for n in sms], WHOLE_RUN,
                f"whole run (Amdahl fit p = {total_fit['p']:.4f})", curve, amdahl_speedup(total_fit, curve))
    plot_series(ax_speedup, sorted(cycle_speedup), [cycle_speedup[n] for n in sorted(cycle_speedup)], CYCLES,
                "photon phase in cycles (clock removed)")
    styled_axes(ax_speedup, f"Strong scaling, FP{precision}: the same N = {points[0]['photons']:.2e} on n SMs",
                "SMs (n)", "speedup S(n) = T(1) / T(n)")
    ax_speedup.legend(frameon=False, fontsize=9, labelcolor=TEXT)
    ax_clock.plot(sms, [clocks.get(n) for n in sms], color=PHOTON_PHASE, linewidth=2, marker="o", markersize=6)
    styled_axes(ax_clock, "SM clock during the run (median)", "SMs (n)", "MHz")

    data = {"precision": precision, "photons": points[0]["photons"], "sms": sms,
            "photon_time": photons, "whole_run_time": totals, "kernel_cycles": cycles, "sm_clock_mhz": clocks,
            "speedup_photon_phase": speedup, "speedup_whole_run": total_speedup, "speedup_cycles": cycle_speedup,
            "amdahl_photon_phase": photon_fit, "amdahl_whole_run": total_fit, "serial_fraction_measured": measured_s}
    save_figure(fig, folder, name, data)
    return data


# ============================================================ weak scaling (Gustafson)

def analyse_weak(points, precision, folder, name):
    photons = medians_by(points, by_sms, photon_time)
    totals = medians_by(points, by_sms, whole_run_time)
    serial = medians_by(points, by_sms, serial_time)
    sms = sorted(photons)
    if len(sms) < 3 or 1 not in sms:
        return None
    scaled = {n: n * photons[1] / photons[n] for n in sms}
    scaled_total = {n: (serial[1] + n * photons[1]) / totals[n] for n in sms}
    efficiency = {n: photons[1] / photons[n] for n in sms}
    photon_fit = fit_gustafson(sms, [scaled[n] for n in sms])
    total_fit = fit_gustafson(sms, [scaled_total[n] for n in sms])
    measured_s_prime = serial[max(sms)] / totals[max(sms)]

    fig, (ax_time, ax_speedup) = plt.subplots(1, 2, figsize=(12, 4.8))
    ax_time.axhline(photons[1], color=IDEAL, linewidth=1, linestyle=":", label="ideal: constant time")
    plot_series(ax_time, sms, [photons[n] for n in sms], PHOTON_PHASE, "photon phase")
    plot_series(ax_time, sms, [totals[n] for n in sms], WHOLE_RUN, "whole run")
    ax_time.set_ylim(bottom=0)
    styled_axes(ax_time, f"Weak scaling, FP{precision}: N grows with the SMs", "SMs (n)", "time (s)")
    ax_time.legend(frameon=False, fontsize=9, labelcolor=TEXT)
    curve = np.linspace(1, max(sms), 200)
    ax_speedup.plot(curve, curve, color=IDEAL, linewidth=1, linestyle=":", label="ideal: S = n")
    plot_series(ax_speedup, sms, [scaled[n] for n in sms], PHOTON_PHASE, "photon phase: n · T(1, W) / T(n, nW)")
    plot_series(ax_speedup, sms, [scaled_total[n] for n in sms], WHOLE_RUN,
                f"whole run: (serial + n · photons(1)) / T(n)  (Gustafson fit s′ = {total_fit['s_prime']:.4f})",
                curve, curve - total_fit["s_prime"] * (curve - 1))
    styled_axes(ax_speedup, "Scaled speedup: one SM's time for the n-fold problem / n SMs' time", "SMs (n)",
                "scaled speedup")
    ax_speedup.legend(frameon=False, fontsize=9, labelcolor=TEXT)

    data = {"precision": precision, "sms": sms, "photons": {p["sms"]: p["photons"] for p in points},
            "photon_time": photons, "whole_run_time": totals, "scaled_speedup_photon_phase": scaled,
            "scaled_speedup_whole_run": scaled_total, "efficiency_photon_phase": efficiency,
            "gustafson_photon_phase": photon_fit, "gustafson_whole_run": total_fit,
            "serial_fraction_measured_on_all_sms": measured_s_prime}
    save_figure(fig, folder, name, data)
    return data


def plot_strong_vs_weak(strong, weak, precision, folder):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    curve = np.linspace(1, ALL_SMS, 200)
    ax.plot(curve, curve, color=IDEAL, linewidth=1, linestyle=":", label="ideal")
    sms = strong["sms"]
    plot_series(ax, sms, [strong["speedup_whole_run"][n] for n in sms], WHOLE_RUN,
                "strong (Amdahl): the same problem, more SMs")
    sms = weak["sms"]
    plot_series(ax, sms, [weak["scaled_speedup_whole_run"][n] for n in sms], PHOTON_PHASE,
                "weak (Gustafson): a bigger problem with more SMs")
    styled_axes(ax, f"Strong vs weak scaling, FP{precision}, whole run", "SMs (n)", "speedup")
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT)
    save_figure(fig, folder, f"strong_vs_weak_fp{precision}", {"strong": strong["speedup_whole_run"],
                                                               "weak": weak["scaled_speedup_whole_run"]})


def plot_amdahl_ceiling(small, large, weak, precision, folder):
    fig, ax = plt.subplots(figsize=(8, 5.2))
    curve = np.linspace(1, ALL_SMS, 200)
    ax.plot(curve, curve, color=IDEAL, linewidth=1, linestyle=":", label="ideal")
    sms = weak["sms"]
    plot_series(ax, sms, [weak["scaled_speedup_whole_run"][n] for n in sms], PHOTON_PHASE,
                "weak (Gustafson): N grows with the SMs")
    sms = large["sms"]
    plot_series(ax, sms, [large["speedup_whole_run"][n] for n in sms], CYCLES,
                f"strong (Amdahl), N = {large['photons']:.1e}: p = {large['amdahl_whole_run']['p']:.4f}")
    sms = small["sms"]
    fit = small["amdahl_whole_run"]
    plot_series(ax, sms, [small["speedup_whole_run"][n] for n in sms], WHOLE_RUN,
                f"strong (Amdahl), N = {small['photons']:.0e}: p = {fit['p']:.3f}, ceiling 1/s = {fit['ceiling']:.1f}",
                curve, amdahl_speedup(fit, curve))
    styled_axes(ax, f"The same GPU, three questions (FP{precision}, whole run)", "SMs (n)", "speedup")
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT)
    save_figure(fig, folder, f"amdahl_ceiling_fp{precision}",
                {"weak": weak["scaled_speedup_whole_run"], "strong_large_n": large["speedup_whole_run"],
                 "strong_small_n": small["speedup_whole_run"], "amdahl_small_n": fit})


# ============================================================ p(N)

def analyse_problem_size(points, folder):
    def by_size_and_sms(point):
        return point["photons"], point["sms"]
    photons = medians_by(points, by_size_and_sms, photon_time)
    totals = medians_by(points, by_size_and_sms, whole_run_time)
    sizes = sorted({size for size, sms in photons if (size, 1) in photons and (size, ALL_SMS) in photons})
    if not sizes:
        return None
    p_photons = {n: parallel_fraction(photons[(n, 1)] / photons[(n, ALL_SMS)]) for n in sizes}
    p_total = {n: parallel_fraction(totals[(n, 1)] / totals[(n, ALL_SMS)]) for n in sizes}
    fig, ax = plt.subplots(figsize=(7.5, 5))
    plot_series(ax, sizes, [p_photons[n] for n in sizes], PHOTON_PHASE, "photon phase")
    plot_series(ax, sizes, [p_total[n] for n in sizes], WHOLE_RUN, "whole run")
    ax.set_xscale("log")
    ax.axhline(1, color=IDEAL, linewidth=1, linestyle=":")
    styled_axes(ax, "Amdahl's parallel fraction p grows with N (FP32, from 1 and 36 SMs)", "photons N", "p")
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT)
    data = {"sizes": sizes, "p_photon_phase": p_photons, "p_whole_run": p_total,
            "photon_time": {f"{n}_{s}": t for (n, s), t in photons.items()},
            "whole_run_time": {f"{n}_{s}": t for (n, s), t in totals.items()}}
    save_figure(fig, folder, "problem_size", data)
    return data


# ============================================================ devices

def analyse_devices(points, folder):
    def by_binary(point):
        return f"{point['processor']}{point['precision']}"
    speeds = medians_by(points, by_binary, lambda point: point["SUMMARY"]["PHOTONS_PER_SECOND"])
    totals = medians_by(points, by_binary, whole_run_time)
    binaries = [b for b in ("cpu64", "cpu32", "gpu64", "gpu32") if b in speeds]
    if not binaries:
        return None
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.barh(binaries, [speeds[b] for b in binaries], color=PHOTON_PHASE, height=0.5)
    ax.set_xscale("log")
    for row, binary in enumerate(binaries):
        ax.text(speeds[binary] * 1.15, row, f"{speeds[binary]:.2g}", va="center", color=TEXT, fontsize=9)
    ax.invert_yaxis()
    styled_axes(ax, "Photons per second, photon phase (CPU: one core; GPU: all 36 SMs)", "photons / s", "")
    def same_precision_cpu(binary):
        return "cpu" + binary[3:]
    data = {"binaries": binaries, "photons_per_second": speeds, "whole_run_time": totals,
            "speedup_photon_phase": {b: speeds[b] / speeds[same_precision_cpu(b)] for b in binaries
                                     if same_precision_cpu(b) in speeds},
            "speedup_whole_run": {b: totals[same_precision_cpu(b)] / totals[b] for b in binaries
                                  if same_precision_cpu(b) in totals}}
    save_figure(fig, folder, "devices", data)
    return data


# ============================================================ the report

def report_lines(results):
    lines = ["# Scaling analysis (the laptop: RTX 5070 Laptop GPU, 36 SMs; CPU baseline: one core)", ""]
    for precision in (64, 32):
        for strong in (results.get(f"strong-gpu{precision}"), results.get(f"strong-gpu{precision}-n1e+08")):
            if not strong:
                continue
            photon, total = strong["amdahl_photon_phase"], strong["amdahl_whole_run"]
            lines += [f"## Strong scaling, FP{precision} (N = {strong['photons']:.2e})", "",
                      "| SMs | photon phase (s) | whole run (s) | speedup, photons | speedup, whole run | "
                      "speedup, cycles | SM clock (MHz) |", "|---|---|---|---|---|---|---|"]
            for n in strong["sms"]:
                lines.append(f"| {n} | {strong['photon_time'][n]:.3f} | {strong['whole_run_time'][n]:.3f} | "
                             f"{strong['speedup_photon_phase'][n]:.2f} | {strong['speedup_whole_run'][n]:.2f} | "
                             f"{strong['speedup_cycles'].get(n, float('nan')):.2f} | "
                             f"{strong['sm_clock_mhz'].get(n) or float('nan'):.0f} |")
            lines += ["", f"- Amdahl fit, photon phase: p = {photon['p']:.5f}, s = {photon['s']:.5f}, "
                          f"ceiling 1/s = {photon['ceiling']:.0f}",
                      f"- Amdahl fit, whole run: p = {total['p']:.5f}, s = {total['s']:.5f}, "
                      f"ceiling 1/s = {total['ceiling']:.0f}",
                      f"- serial fraction measured from the phases (setup + zero + copy + save) / T(1): "
                      f"{strong['serial_fraction_measured']:.5f}", ""]
        weak = results.get(f"weak-gpu{precision}")
        if weak:
            lines += [f"## Weak scaling, FP{precision}", "", "| SMs | N | photon phase (s) | whole run (s) | "
                      "scaled speedup, photons | scaled speedup, whole run | efficiency |",
                      "|---|---|---|---|---|---|---|"]
            for n in weak["sms"]:
                lines.append(f"| {n} | {weak['photons'][n]:.2e} | {weak['photon_time'][n]:.3f} | "
                             f"{weak['whole_run_time'][n]:.3f} | {weak['scaled_speedup_photon_phase'][n]:.2f} | "
                             f"{weak['scaled_speedup_whole_run'][n]:.2f} | {weak['efficiency_photon_phase'][n]:.3f} |")
            lines += ["", "- scaled speedup, whole run = (serial + n · photon time on 1 SM) / T(n): the time one SM "
                          "would take for the n-fold problem, over the time n SMs take (Gustafson's definition)",
                      f"- Gustafson fit, whole run: s′ = {weak['gustafson_whole_run']['s_prime']:.5f}",
                      f"- serial share measured on 36 SMs: {weak['serial_fraction_measured_on_all_sms']:.5f}", ""]
    size = results.get("problem-size-gpu32")
    if size:
        lines += ["## p(N), FP32 (from 1 and 36 SMs)", "", "| N | p, photon phase | p, whole run |", "|---|---|---|"]
        lines += [f"| {n:.0e} | {size['p_photon_phase'][n]:.5f} | {size['p_whole_run'][n]:.5f} |"
                  for n in size["sizes"]]
        lines.append("")
    devices = results.get("devices")
    if devices:
        lines += ["## Devices (N = 10⁸; the GPU: all 36 SMs; the CPU: one core; speedups at the same precision)", "",
                  "| binary | photons/s (photon phase) | whole run (s) | speedup, photon phase | speedup, whole run |",
                  "|---|---|---|---|---|"]
        for binary in devices["binaries"]:
            lines.append(f"| {binary} | {devices['photons_per_second'][binary]:.3g} | "
                         f"{devices['whole_run_time'][binary]:.3f} | "
                         f"{devices['speedup_photon_phase'].get(binary, float('nan')):.1f}× | "
                         f"{devices['speedup_whole_run'].get(binary, float('nan')):.1f}× |")
    return lines


ANALYSES = {"strong": analyse_strong, "weak": analyse_weak}


def analyse(label, folder, output_folder):
    points = read_points(folder)
    if not points:
        return None
    if label == "devices":
        return analyse_devices(points, output_folder)
    if label.startswith("problem-size"):
        return analyse_problem_size(points, output_folder)
    kind, binary, *size = label.split("-")
    name = f"{kind}_fp{binary[3:]}" + (f"_{size[0]}" if size else "")
    return ANALYSES[kind](points, int(binary[3:]), output_folder, name)


def main():
    parser = argparse.ArgumentParser(description="Amdahl and Gustafson from the sweeps (the latest of each kind)")
    parser.parse_args()
    output_folder = os.path.join(RUNS_FOLDER, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "-scaling")
    os.makedirs(output_folder)
    results = {}
    for label in SWEEP_LABELS:
        folder = latest_sweep_folder(label)
        if folder:
            results[label] = analyse(label, folder, output_folder)
    for precision in (64, 32):
        strong, weak = results.get(f"strong-gpu{precision}"), results.get(f"weak-gpu{precision}")
        if strong and weak:
            plot_strong_vs_weak(strong, weak, precision, output_folder)
        small = results.get(f"strong-gpu{precision}-n1e+08")
        if small and strong and weak:
            plot_amdahl_ceiling(small, strong, weak, precision, output_folder)
    lines = report_lines({label: result for label, result in results.items() if result})
    with open(os.path.join(output_folder, "report.md"), "w") as file:
        file.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nfigures in {os.path.relpath(output_folder, engine.HERE)}")


if __name__ == "__main__":
    main()
