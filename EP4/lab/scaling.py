import json
import os
import statistics
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PHOTON_PHASE = "#2a78d6"
WHOLE_RUN = "#eb6834"
CYCLES = "#1baf7a"
IDEAL = "#8a8985"
TEXT = "#52514e"
GRID = "#e4e3df"


# ============================================================ what a point measured

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


PHASES = (("setup", "SETUP_TIME"), ("zeroing", "ZERO_TIME"), ("photons", "PHOTONS_TIME"),
          ("copy back", "COPY_TIME"), ("save", "SAVE_TIME"))
PHASE_COLORS = {"setup": "#b9b8b3", "zeroing": "#d9d8d4", "photons": "#2a78d6", "copy back": "#f4a582",
                "save": "#eb6834", "other": "#e4e3df"}


def phase_time(key):
    return lambda point: point["SUMMARY"].get(key)


def medians_by(points, key, measure):
    groups = defaultdict(list)
    for point in points:
        value = measure(point)
        if value is not None:
            groups[key(point)].append(value)
    return {group: statistics.median(values) for group, values in sorted(groups.items())}


def by_sms(point):
    return point["sms"]


def photon_clock_mhz(cycles, photons):
    return {n: cycles[n] / photons[n] / 1e6 for n in photons if n in cycles}


def at(series, n):
    return series[n] if n in series else series[str(n)]


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


def parallel_fraction(speedup, units):
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


def plot_photon_clock(ax, sms, clock, xlim=None):
    ax.plot(sms, [at(clock, n) for n in sms], color=PHOTON_PHASE, linewidth=2, marker="o", markersize=6)
    ax.set_ylim(0, 1.15 * max(at(clock, n) for n in sms))
    if xlim:
        ax.set_xlim(0, xlim)
    styled_axes(ax, "SM clock during the photons (cycles ÷ time)", "SMs (n)", "MHz")


# ============================================================ the phases of a run

def phases_by_sms(points):
    medians = {name: medians_by(points, by_sms, phase_time(key)) for name, key in PHASES}
    totals = medians_by(points, by_sms, whole_run_time)
    phases = {}
    for n, whole in totals.items():
        times = {name: medians[name].get(n) or 0.0 for name, _ in PHASES}
        phases[n] = dict(times, other=max(0.0, whole - sum(times.values())), whole=whole)
    return phases


def plot_phases(phases, precision, photons, folder):
    shown = [min(phases), max(phases)]
    names = [name for name, _ in PHASES] + ["other"]
    fig, ax = plt.subplots(figsize=(11, 3.2))
    for row, n in enumerate(shown):
        left = 0.0
        for name in names:
            share = phases[n][name] / phases[n]["whole"]
            ax.barh(row, share, left=left, color=PHASE_COLORS[name], height=0.55, edgecolor="white",
                    label=name if row == 0 else None)
            if share > 0.06:
                ax.text(left + share / 2, row, f"{name}\n{phases[n][name]:.1f} s", ha="center", va="center",
                        fontsize=9, color="white" if name in ("photons", "save") else TEXT)
            left += share
    ax.set_yticks(range(len(shown)), [f"{n} SM" + ("s" if n > 1 else "") + f"\n{phases[n]['whole']:.1f} s"
                                      for n in shown])
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    styled_axes(ax, f"Where the whole run goes, FP{precision}, N = {photons:.1e}: share of each phase",
                "share of the whole run", "")
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT, ncol=len(names), loc="upper center",
              bbox_to_anchor=(0.5, -0.32))
    save_figure(fig, folder, f"phases_fp{precision}", {str(n): phases[n] for n in phases})


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
    clock_during_photons = photon_clock_mhz(cycles, photons)
    phases = phases_by_sms(points)
    plot_phases(phases, precision, points[0]["photons"], folder)

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
    plot_photon_clock(ax_clock, sms, clock_during_photons)

    data = {"precision": precision, "photons": points[0]["photons"], "sms": sms,
            "photon_time": photons, "whole_run_time": totals, "kernel_cycles": cycles, "sm_clock_mhz": clocks,
            "sm_clock_during_photons_mhz": clock_during_photons, "phases": phases,
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
    curve = np.linspace(1, max(strong["sms"]), 200)
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


CEILING_REACH = 3


def photon_fit_line(fit, curve):
    if fit["s"] <= 0:
        return (f"photon phase, in seconds; fit p = {fit['p']:.4f} > 1, not extended",)
    return f"photon phase, in seconds; Amdahl fit p = {fit['p']:.4f}", curve, amdahl_speedup(fit, curve)


def plot_amdahl_ceiling(strong, precision, folder):
    fig, (ax, ax_clock) = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [2, 1]})
    sms = strong["sms"]
    photon_fit, whole_fit = strong["amdahl_photon_phase"], strong["amdahl_whole_run"]
    reach = CEILING_REACH * max(sms)
    curve = np.linspace(1, reach, 300)
    ax.plot(curve, curve, color=IDEAL, linewidth=1, linestyle=":", label="ideal: S = n")
    plot_series(ax, sms, [at(strong["speedup_photon_phase"], n) for n in sms], PHOTON_PHASE,
                *photon_fit_line(photon_fit, curve))
    plot_series(ax, sms, [at(strong["speedup_cycles"], n) for n in sms], CYCLES,
                "photon phase, in cycles (the clock removed)")
    plot_series(ax, sms, [at(strong["speedup_whole_run"], n) for n in sms], WHOLE_RUN,
                f"whole run, in seconds; Amdahl fit p = {whole_fit['p']:.4f}", curve, amdahl_speedup(whole_fit, curve))
    ax.axhline(whole_fit["ceiling"], color=WHOLE_RUN, linewidth=1, linestyle=":",
               label=f"whole run's ceiling 1/s = {whole_fit['ceiling']:.0f}, however many SMs")
    ax.axvline(max(sms), color=GRID, linewidth=1.5)
    ax.set_xlim(0, reach)
    ax.set_ylim(0, reach)
    styled_axes(ax, f"Amdahl's ceiling, FP{precision}, N = {strong['photons']:.1e}: measured up to {max(sms)} SMs, "
                    f"the fits beyond", "SMs (n)", "speedup S(n) = T(1) / T(n)")
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT, loc="upper left")
    plot_photon_clock(ax_clock, sms, strong["sm_clock_during_photons_mhz"], xlim=max(sms) * 1.05)
    save_figure(fig, folder, f"amdahl_ceiling_fp{precision}",
                {"photon_phase": strong["speedup_photon_phase"], "cycles": strong["speedup_cycles"],
                 "whole_run": strong["speedup_whole_run"], "amdahl_photon_phase": photon_fit,
                 "amdahl_whole_run": whole_fit, "sm_clock_during_photons_mhz": strong["sm_clock_during_photons_mhz"]})


# ============================================================ p(N)

def analyse_problem_size(points, folder):
    def by_size_and_sms(point):
        return point["photons"], point["sms"]
    photons = medians_by(points, by_size_and_sms, photon_time)
    totals = medians_by(points, by_size_and_sms, whole_run_time)
    every = max(point["sms"] for point in points)
    sizes = sorted({size for size, sms in photons if (size, 1) in photons and (size, every) in photons})
    if not sizes:
        return None
    p_photons = {n: parallel_fraction(photons[(n, 1)] / photons[(n, every)], every) for n in sizes}
    p_total = {n: parallel_fraction(totals[(n, 1)] / totals[(n, every)], every) for n in sizes}
    fig, ax = plt.subplots(figsize=(7.5, 5))
    plot_series(ax, sizes, [p_photons[n] for n in sizes], PHOTON_PHASE, "photon phase")
    plot_series(ax, sizes, [p_total[n] for n in sizes], WHOLE_RUN, "whole run")
    ax.set_xscale("log")
    ax.axhline(1, color=IDEAL, linewidth=1, linestyle=":")
    styled_axes(ax, f"Amdahl's parallel fraction p grows with N (FP32, from 1 and {every} SMs)", "photons N", "p")
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT)
    data = {"sizes": sizes, "p_photon_phase": p_photons, "p_whole_run": p_total,
            "photon_time": {f"{n}_{s}": t for (n, s), t in photons.items()},
            "whole_run_time": {f"{n}_{s}": t for (n, s), t in totals.items()}}
    save_figure(fig, folder, "problem_size", data)
    return data
