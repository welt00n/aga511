"""EP3: Buffon's needle on the GPU and the CPU (AGA0511).

    python buffon.py                         GPU, double (FP64), for 10 minutes
    python buffon.py --precision 32          float (FP32); 16 = half (FP16)
    python buffon.py --cpu                   EP1 on one CPU core
    python buffon.py --budget 1h             run longer (also 90s, 30m, 2h)
    python buffon.py --n 1e9                 one N only, no sweep
    python buffon.py --info                  what the GPU is
    python buffon.py --plot runs/<folder>    redraw a finished run
    python buffon.py --report runs/<a> runs/<b> ...   compare finished runs
"""
import argparse
import csv
import json
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# LINK_TO_COLLAB="https://colab.research.google.com/drive/1IEIpwo6pjTRLXpqj3WFxqWebQK2ln2Py"
HERE = Path(__file__).resolve().parent
ENGINE = HERE / "buffon"
RUNS = HERE / "runs"

TARGET_DECIMALS = 14
TARGET_ERROR = 10.0 ** -TARGET_DECIMALS
DECIMALS = list(range(3, TARGET_DECIMALS + 1))
PRECISION_DECIMALS = {64: 14, 32: 6, 16: 3}
SIGNIFICANT_DIGITS = {64: 16, 32: 7, 16: 3}

LIFETIME_YEARS = 80
SECONDS_PER_YEAR = 365.25 * 24 * 3600
AGE_OF_UNIVERSE_YEARS = 1.38e10
REFERENCE_TIMES = [(60, "1 minute"), (3600, "1 hour"), (86400, "1 day"), (SECONDS_PER_YEAR, "1 year"),
                   (LIFETIME_YEARS * SECONDS_PER_YEAR, f"{LIFETIME_YEARS} years (a lifetime)"),
                   (AGE_OF_UNIVERSE_YEARS * SECONDS_PER_YEAR, "age of the universe")]

FIRST_N = 10**4
STEP = 10**0.5
LARGEST_N = 9 * 10**18
GPU_CHECK_N = 10**7
CPU_CHECK_N = 10**6
HELD_OUT_FACTOR = 1000
STATISTICAL_LIMIT = 1.5
FLOOR_LIMIT = 3
TIMEABLE_LOOP_SECONDS = 0.1
LINEAR_TOLERANCE = 0.05
TIME_RESOLUTION = 1e-6
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 0

P_CROSS = 2 / np.pi
K_THEORY = np.sqrt(4 * (1 - P_CROSS) / P_CROSS**3)

COLUMNS = {
    "device": str, "precision": int, "n": int, "seed": int, "sms": int, "sms_used": int,
    "blocks": int, "threads": int, "chunk_size": int, "pi": float, "error": float, "sigma": float,
    "time": float, "loop_time": float, "cycles": int, "clock_mhz": int,
}

SPEEDUPS = (("Total time, fixed clock", "time_fixed"),
            ("Total time, wall clock", "time"),
            ("Needle loop only, fixed clock", "loop_time_fixed"))
SPEEDUP_NAMES = {SPEEDUPS[0][0]: "Whole run", SPEEDUPS[2][0]: "Needle loop only"}


# ============================================================ command line

def needles(text):
    value = int(float(text))
    if value < 1 or value > LARGEST_N:
        raise argparse.ArgumentTypeError(f"must be a number of needles from 1 to {LARGEST_N:.1e}")
    return value

def duration(text):
    units = {"s": 1, "m": 60, "h": 3600}
    seconds = float(text[:-1]) * units[text[-1]] if text[-1] in units else float(text)
    if seconds <= 0:
        raise argparse.ArgumentTypeError("must be a positive duration")
    return seconds

def repetitions(text):
    value = int(text)
    if value < 2:
        raise argparse.ArgumentTypeError("must be at least 2: the uncertainties need a spread between runs")
    return value

def seed(text):
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be zero or positive")
    return value

def parse_args():
    parser = argparse.ArgumentParser(description="Buffon's needle on the GPU and the CPU (AGA0511 EP3)")
    parser.add_argument("--cpu", action="store_true", help="run EP1 on one CPU core instead of the GPU")
    parser.add_argument("--precision", type=int, choices=[64, 32, 16], default=64,
                        help="64 = double, 32 = float, 16 = half (default: 64)")
    parser.add_argument("--budget", type=duration, default=600,
                        help="how long the experiment may run: 600, 90s, 10m, 2h (default: 10m)")
    parser.add_argument("--n", type=needles, help="run only this N instead of the sweep, e.g. 1e9")
    parser.add_argument("--repetitions", type=repetitions,
                        help="seeds per N (default: 3 on the GPU, each on every SM count; 20 on the CPU)")
    parser.add_argument("--seed", type=seed, default=42,
                        help="seed of the first run; each run takes the next one (default: 42)")
    other = parser.add_mutually_exclusive_group()
    other.add_argument("--info", action="store_true", help="describe the GPU and exit")
    other.add_argument("--plot", type=Path, metavar="FOLDER", help="redraw the graphs of a finished run")
    other.add_argument("--report", type=Path, nargs="+", metavar="FOLDER", help="compare finished runs")
    return parser.parse_args()


# ============================================================ the engine (buffon.cu)

def compile_engine():
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    command = [nvcc, "buffon.cu", "-o", "buffon", "-O2", "-arch=native"]
    result = subprocess.run(command, cwd=HERE, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Error compiling the CUDA engine:\n{result.stderr}")

def parse_key_values(text):
    values = {}
    for item in text.split():
        key, value = item.split("=", 1)
        values[key] = value
    return values

def run_engine(*flags):
    command = [str(ENGINE)] + [str(flag) for flag in flags]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Error running the engine:\n{result.stderr}")
    return parse_key_values(result.stdout)

def engine_flags(n, seed, precision, sms, cpu):
    flags = ["-n", n, "-s", seed % 2**32, "-p", precision]
    if cpu:
        return flags + ["-c"]
    return flags + ["-m", sms]

def throw_needles(n, seed, precision, sms=0, cpu=False):
    output = run_engine(*engine_flags(n, seed, precision, sms, cpu))
    row = {}
    for column, kind in COLUMNS.items():
        row[column] = kind(output[column.upper()])
    return row

def read_gpu_info():
    return run_engine("-i")

def read_cpu_name():
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return "unknown CPU"

def sm_counts(total):
    counts = []
    count = 1
    while count < total:
        counts.append(count)
        count *= 2
    return counts + [total]


# ============================================================ run folders

def new_run_folder(name, config):
    folder = RUNS / f"{datetime.now():%Y-%m-%d_%H-%M-%S}-{name}"
    folder.mkdir(parents=True)
    (folder / "config.json").write_text(json.dumps(config, indent=2))
    with open(folder / "results.csv", "w", newline="") as file:
        csv.DictWriter(file, fieldnames=list(COLUMNS)).writeheader()
    return folder

def save_result(folder, row):
    with open(folder / "results.csv", "a", newline="") as file:
        csv.DictWriter(file, fieldnames=list(COLUMNS)).writerow(row)

def load_run(folder):
    config = json.loads((folder / "config.json").read_text())
    results = pd.read_csv(folder / "results.csv")
    return config, results


# ============================================================ the experiment

def is_cpu(config):
    return config["device"] == "cpu"

def cost_model(serial_seconds, seconds_per_needle_on_one_sm):
    return {"serial": serial_seconds, "needle_time": seconds_per_needle_on_one_sm}

def measure_cpu_cost(config):
    started = time.perf_counter()
    row = throw_needles(CPU_CHECK_N, config["seed"], config["precision"], cpu=True)
    wall = time.perf_counter() - started
    return cost_model(wall - row["loop_time"], row["loop_time"] / CPU_CHECK_N)

def check_same_pi(one, full, all_sms, seed):
    print(f"Check: seed {seed} gives pi = {one['pi']:.15f} on 1 SM and {full['pi']:.15f} on {all_sms} SMs", end=" ")
    if one["pi"] != full["pi"]:
        raise SystemExit("-> DIFFERENT: the engine is not sharing the work correctly")
    print("-> identical, OK")

def measure_gpu_cost(config):
    seed = config["seed"]
    precision = config["precision"]
    all_sms = config["sm_counts"][-1]
    throw_needles(GPU_CHECK_N, seed, precision)
    started = time.perf_counter()
    full = throw_needles(GPU_CHECK_N, seed, precision, sms=all_sms)
    wall = time.perf_counter() - started
    one = throw_needles(GPU_CHECK_N, seed, precision, sms=1)
    check_same_pi(one, full, all_sms, seed)
    return cost_model(wall - full["loop_time"], one["loop_time"] / GPU_CHECK_N)

def measure_cost(config):
    if is_cpu(config):
        return measure_cpu_cost(config)
    return measure_gpu_cost(config)

def remeasure_cost(rows, step_seconds):
    loop_seconds = 0.0
    sm_seconds = 0.0
    needles = 0
    for row in rows:
        loop_seconds += row["loop_time"]
        sm_seconds += row["loop_time"] * row["sms"]
        needles += row["n"]
    return cost_model((step_seconds - loop_seconds) / len(rows), sm_seconds / needles)

def step_cost(cost, n, config):
    one_seed = 0.0
    for sms in config["sm_counts"]:
        one_seed += cost["serial"] + cost["needle_time"] * n / sms
    return config["repetitions"] * one_seed

def sweep(config):
    if config["n"]:
        return [config["n"]]
    values = []
    n = FIRST_N
    while n <= LARGEST_N:
        values.append(int(round(n)))
        n *= STEP
    return values

def plan(cost, config):
    planned = []
    total = 0.0
    for n in sweep(config):
        seconds = step_cost(cost, n, config)
        if planned and total + seconds > config["budget"]:
            break
        planned.append(n)
        total += seconds
    return planned, total

def print_plan(planned, total, config):
    runs = len(planned) * config["repetitions"] * len(config["sm_counts"])
    print(f"Plan: N = {planned[0]:.1e} ... {planned[-1]:.1e} ({len(planned)} values) "
          f"x {config['repetitions']} seeds x SM counts {config['sm_counts']} = {runs} runs, "
          f"about {human_time(total)} (budget {human_time(config['budget'])})")

def warn_if_sms_differ(row):
    if row["sms_used"] != row["sms"]:
        print(f"  warning: asked for {row['sms']} SMs but {row['sms_used']} worked")

def run_step(n, seed, config, folder):
    rows = []
    for repetition in range(config["repetitions"]):
        for sms in config["sm_counts"]:
            row = throw_needles(n, seed, config["precision"], sms, is_cpu(config))
            warn_if_sms_differ(row)
            save_result(folder, row)
            rows.append(row)
            seed += 1
    return rows, seed

def run_sweep(config, folder, cost):
    seed = config["seed"]
    started = time.perf_counter()
    for step, n in enumerate(sweep(config)):
        elapsed = time.perf_counter() - started
        if step > 0 and elapsed + step_cost(cost, n, config) > config["budget"]:
            return
        step_started = time.perf_counter()
        rows, seed = run_step(n, seed, config, folder)
        step_seconds = time.perf_counter() - step_started
        cost = remeasure_cost(rows, step_seconds)
        print(f"N = {n:.2e}: {len(rows)} runs in {step_seconds:.1f} s "
              f"(elapsed {time.perf_counter() - started:.0f} s of {config['budget']:.0f} s)")

def run_experiment(config, folder):
    cost = measure_cost(config)
    planned, total = plan(cost, config)
    print_plan(planned, total, config)
    try:
        run_sweep(config, folder, cost)
    except KeyboardInterrupt:
        print("\nStopped by the user: analysing the runs saved so far.")


# ============================================================ uncertainties

def chi2_scale(chi2, dof):
    if dof <= 0:
        return 1.0
    return max(1.0, float(np.sqrt(chi2 / dof)))

def weighted_line_fit(x, y, sigma):
    weight = 1 / sigma**2
    s = np.sum(weight)
    sx = np.sum(weight * x)
    sy = np.sum(weight * y)
    sxx = np.sum(weight * x * x)
    sxy = np.sum(weight * x * y)
    determinant = s * sxx - sx * sx
    slope = (s * sxy - sx * sy) / determinant
    intercept = (sxx * sy - sx * sxy) / determinant
    chi2 = float(np.sum(weight * (y - slope * x - intercept) ** 2))
    dof = len(x) - 2
    scale = chi2_scale(chi2, dof)
    return {"slope": float(slope), "intercept": float(intercept),
            "slope_uncertainty": float(np.sqrt(s / determinant)) * scale,
            "intercept_uncertainty": float(np.sqrt(sxx / determinant)) * scale,
            "chi2": chi2, "dof": dof}

def bootstrap_k_uncertainty(k_squared):
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    samples = generator.choice(k_squared, size=(BOOTSTRAP_SAMPLES, len(k_squared)))
    return float(np.std(np.sqrt(samples.mean(axis=1)), ddof=1))

def ratio_with_uncertainty(measured, measured_uncertainty, predicted, predicted_uncertainty):
    ratio = measured / predicted
    return ratio, ratio * np.sqrt((measured_uncertainty / measured) ** 2 + (predicted_uncertainty / predicted) ** 2)

def weighted_mean(measurements):
    values = np.array([value for value, _ in measurements])
    weight = 1 / np.maximum(np.array([uncertainty for _, uncertainty in measurements]), 1e-12) ** 2
    return float(np.sum(weight * values) / np.sum(weight)), float(np.sqrt(1 / np.sum(weight)))

def floored(uncertainties):
    return np.maximum(uncertainties, TIME_RESOLUTION)


# ============================================================ formatting

def pdg_rounded_uncertainty(uncertainty):
    exponent = int(np.floor(np.log10(uncertainty)))
    leading_digits = int(uncertainty / 10.0 ** (exponent - 2))
    if leading_digits >= 950:
        return 10.0 ** (exponent + 1), 2
    if leading_digits >= 355:
        return round(uncertainty, -exponent), 1
    return round(uncertainty, 1 - exponent), 2

def uncertainty_decimals(uncertainty, significant_digits):
    return significant_digits - 1 - int(np.floor(np.log10(uncertainty)))

def fixed_point(value, uncertainty, plus_minus):
    rounded, digits = pdg_rounded_uncertainty(uncertainty)
    decimals = uncertainty_decimals(rounded, digits)
    if decimals < 0:
        return f"{round(value, decimals):.0f}{plus_minus}{rounded:.0f}"
    return f"{value:.{decimals}f}{plus_minus}{rounded:.{decimals}f}"

def format_measurement(value, uncertainty, tex=False):
    plus_minus = r" \pm " if tex else " ± "
    if not np.isfinite(uncertainty) or uncertainty <= 0:
        return f"{value:.4g}"
    if 1e-3 <= abs(value) < 1e5 or abs(value) < uncertainty:
        return fixed_point(value, uncertainty, plus_minus)
    exponent = int(np.floor(np.log10(abs(value))))
    scale = 10.0 ** exponent
    body = "(" + fixed_point(value / scale, uncertainty / scale, plus_minus) + ")"
    if tex:
        return body + rf" \times 10^{{{exponent}}}"
    return body + f"e{exponent}"

def human_time(seconds):
    if seconds < 120:
        return f"{seconds:.1f} s"
    if seconds < 2 * 3600:
        return f"{seconds / 60:.0f} min"
    if seconds < 2 * 86400:
        return f"{seconds / 3600:.1f} h"
    years = seconds / SECONDS_PER_YEAR
    if years < 2:
        return f"{seconds / 86400:.0f} days"
    if years < 1e4:
        return f"{years:.0f} years"
    return f"{years:.1e} years"

def tex_number(value, digits=1):
    exponent = int(np.floor(np.log10(abs(value))))
    exact_mantissa = value / 10.0 ** exponent
    mantissa = round(exact_mantissa, digits)
    if mantissa >= 10:
        mantissa /= 10
        exponent += 1
    if abs(exact_mantissa - 1) < 1e-9:
        return rf"10^{{{exponent}}}"
    return rf"{mantissa:.{digits}f} \times 10^{{{exponent}}}"


# ============================================================ fits

def fit_seconds_per_needle(n, loop_seconds, loop_uncertainties):
    timeable = loop_seconds > TIMEABLE_LOOP_SECONDS
    if timeable.sum() < 2:
        return loop_seconds[-1] / n[-1], loop_uncertainties[-1] / n[-1]
    scale = n[timeable].max()
    line = weighted_line_fit(n[timeable] / scale, loop_seconds[timeable], loop_uncertainties[timeable])
    return line["slope"] / scale, line["slope_uncertainty"] / scale

def fit_serial_seconds(times):
    serial = (times["time"] - times["loop_time"]).to_numpy()
    if len(serial) < 2:
        return float(serial[0]), 0.0
    return float(np.mean(serial)), float(np.std(serial, ddof=1) / np.sqrt(len(serial)))

def predicted_seconds_uncertainty(model, n, n_uncertainty):
    return float(np.sqrt(model["a_uncertainty"] ** 2 + (n * model["b_uncertainty"]) ** 2 + (model["b"] * n_uncertainty) ** 2))

def combined_pi(results):
    crossings = float(np.round(2 * results["n"] / results["pi"]).sum())
    needles = float(results["n"].sum())
    fraction = crossings / needles
    estimate = 2 * needles / crossings
    error = estimate * np.sqrt((1 - fraction) / (fraction * needles))
    return {"pi": float(estimate), "pi_uncertainty": float(error), "needles": needles}

def clock_spread(results):
    full = results[results["sms"] == results["sms"].max()]
    long_loops = full[(full["loop_time"] > TIMEABLE_LOOP_SECONDS) & (full["cycles"] > 0)]
    if len(long_loops) < 2:
        return None
    clock = long_loops["cycles"] / long_loops["loop_time"]
    return float(np.std(clock, ddof=1) / np.mean(clock))

def root_mean_square(values):
    return np.sqrt(np.mean(values**2))

def error_table(results):
    by_n = results.groupby("n")
    table = pd.DataFrame({
        "pi": by_n["pi"].mean(),
        "pi_spread": by_n["pi"].std(ddof=1),
        "rmse": by_n["error"].apply(root_mean_square),
        "sigma": by_n["sigma"].mean(),
        "runs": by_n.size(),
    })
    table["rmse_uncertainty"] = table["rmse"] / np.sqrt(2 * table["runs"])
    return table

def fit_power_law(x, y, relative_uncertainty):
    line = weighted_line_fit(np.log(x), np.log(y), relative_uncertainty)
    return {"c": float(np.exp(line["intercept"])), "exponent": line["slope"],
            "exponent_uncertainty": line["slope_uncertainty"], "chi2": line["chi2"], "dof": line["dof"]}

def full_device_times(results):
    full = results[results["sms"] == results["sms"].max()]
    grouped = full.groupby("n")
    times = grouped[["time", "loop_time"]].mean()
    runs = grouped.size()
    times["time_uncertainty"] = (grouped["time"].std(ddof=1) / np.sqrt(runs)).fillna(0)
    times["loop_time_uncertainty"] = (grouped["loop_time"].std(ddof=1) / np.sqrt(runs)).fillna(0)
    return times

def statistical_mask(table):
    statistical = (table["rmse"] < STATISTICAL_LIMIT * table["sigma"]).to_numpy()
    if statistical.sum() >= 2:
        return statistical
    return np.ones(len(table), dtype=bool)

def fit_error(table, results):
    n = table.index.to_numpy(dtype=float)
    mask = statistical_mask(table)
    used = results[results["n"].isin(table.index[mask])]
    k_squared = (used["n"] * used["error"] ** 2).to_numpy()
    k = float(np.sqrt(np.mean(k_squared)))
    fit = {"mask": mask, "k": k, "k_uncertainty": k / np.sqrt(2 * len(k_squared)),
           "k_bootstrap_uncertainty": bootstrap_k_uncertainty(k_squared), "runs_used": len(k_squared)}
    if mask.sum() >= 3:
        rmse = table["rmse"].to_numpy()
        relative = (table["rmse_uncertainty"] / table["rmse"]).to_numpy()
        power = fit_power_law(n[mask], rmse[mask], relative[mask])
        fit["c"] = power["c"]
        fit["alpha"] = power["exponent"]
        fit["alpha_uncertainty"] = power["exponent_uncertainty"]
        fit["chi2"] = power["chi2"]
        fit["dof"] = power["dof"]
    return fit

def fit_loop_exponent(n, loop_seconds, loop_uncertainties):
    timeable = loop_seconds > TIMEABLE_LOOP_SECONDS
    if timeable.sum() < 3:
        return None, None
    power = fit_power_law(n[timeable], loop_seconds[timeable], loop_uncertainties[timeable] / loop_seconds[timeable])
    return power["exponent"], power["exponent_uncertainty"]

def fit_time(times):
    n = times.index.to_numpy(dtype=float)
    loop_seconds = times["loop_time"].to_numpy()
    loop_uncertainties = floored(times["loop_time_uncertainty"].to_numpy())
    b, b_uncertainty = fit_seconds_per_needle(n, loop_seconds, loop_uncertainties)
    loop_start = max(0.0, loop_seconds[0] - b * n[0])
    serial, serial_uncertainty = fit_serial_seconds(times)
    beta, beta_uncertainty = fit_loop_exponent(n, loop_seconds, loop_uncertainties)
    return {"a": max(0.0, serial + loop_start), "a_uncertainty": serial_uncertainty, "b": float(b), "b_uncertainty": float(b_uncertainty),
            "beta": beta, "beta_uncertainty": beta_uncertainty}

def predicted_seconds(model, n):
    return model["a"] + model["b"] * n

def checkpoints(k, k_uncertainty, model):
    points = []
    for decimals in DECIMALS:
        error = 10.0 ** -decimals
        n = (k / error) ** 2
        n_uncertainty = 2 * n * k_uncertainty / k
        points.append({"decimals": decimals, "error": error, "n": float(n), "n_uncertainty": float(n_uncertainty),
                       "seconds": predicted_seconds(model, n),
                       "seconds_uncertainty": predicted_seconds_uncertainty(model, n, n_uncertainty)})
    return points

def check_row(n, in_fit, fit, model, table, times):
    return {"n": float(n), "in_fit": bool(in_fit),
            "error_predicted": fit["k"] / np.sqrt(n), "error_predicted_uncertainty": fit["k_uncertainty"] / np.sqrt(n),
            "error_measured": float(table.loc[n, "rmse"]), "error_measured_uncertainty": float(table.loc[n, "rmse_uncertainty"]),
            "time_predicted": predicted_seconds(model, n), "time_predicted_uncertainty": predicted_seconds_uncertainty(model, n, 0.0),
            "time_measured": float(times.loc[n, "time"]), "time_measured_uncertainty": float(times.loc[n, "time_uncertainty"])}

def prediction_check(results, table, times):
    n = table.index.to_numpy(dtype=float)
    split = n[-1] / HELD_OUT_FACTOR
    small = n <= split
    if small.sum() < 3:
        return None
    fit = fit_error(table[small], results[results["n"] <= split])
    model = fit_time(times[times.index <= split])
    rows = []
    for value in table.index:
        rows.append(check_row(value, value <= split, fit, model, table, times))
    return {"split": float(split), "k": fit["k"], "k_uncertainty": fit["k_uncertainty"], "a": model["a"],
            "a_uncertainty": model["a_uncertainty"], "b": model["b"], "b_uncertainty": model["b_uncertainty"], "rows": rows}

def decimals_limit(precision, bias, floor):
    if floor:
        return int(np.floor(-np.log10(max(bias, TARGET_ERROR))))
    return PRECISION_DECIMALS[precision]

def free_fit_target(fit):
    if "alpha" not in fit:
        return None
    return float((TARGET_ERROR / fit["c"]) ** (1 / fit["alpha"]))

def analyse_errors(results, config):
    table = error_table(results)
    times = full_device_times(results)
    fit = fit_error(table, results)
    model = fit_time(times)
    floor = bool(table["rmse"].iloc[-1] > FLOOR_LIMIT * table["sigma"].iloc[-1])
    bias = float(abs(table["pi"].iloc[-1] - np.pi))
    return {
        "table": table, "times": times, "fit": fit, "model": model, "floor": floor, "bias": bias,
        "bias_uncertainty": float(table["pi_spread"].iloc[-1] / np.sqrt(table["runs"].iloc[-1])),
        "limit": decimals_limit(config["precision"], bias, floor),
        "reached": int(np.floor(-np.log10(table["rmse"].iloc[-1]))),
        "measured_decimals": float(-np.log10(table["rmse"].iloc[-1])),
        "points": checkpoints(fit["k"], fit["k_uncertainty"], model),
        "check": prediction_check(results, table, times),
        "combined": combined_pi(results),
        "clock_spread": clock_spread(results),
    }

# ============================================================ EP1 graphs

def device_scope(config):
    if is_cpu(config):
        return "one CPU core"
    return f"all {config['sm_counts'][-1]} SMs"

def plot_error_points(ax, results, table):
    n = table.index.to_numpy(dtype=float)
    ax.scatter(results["n"], results["error"], s=6, alpha=0.2, color="gray",
               label=r"Real error of each run, $|\hat{\pi} - \pi|$")
    ax.errorbar(n, table["rmse"], yerr=table["rmse_uncertainty"], fmt="o", color="C0", capsize=2,
                label=r"Real error, root mean square over the runs ($\pm 1\sigma$)")
    ax.plot(n, table["sigma"], "s-", markerfacecolor="none", color="C1",
            label=r"Experimental error $\sigma$, estimated from each run alone")
    ax.plot(n, K_THEORY / np.sqrt(n), "k--", linewidth=1, label=rf"Theory: ${K_THEORY:.2f}/\sqrt{{N}}$")

def plot_free_fit(ax, table, fit):
    if "alpha" not in fit:
        return
    n_fit = table.index.to_numpy(dtype=float)[fit["mask"]]
    exponent = format_measurement(fit["alpha"], fit["alpha_uncertainty"], tex=True)
    ax.plot(n_fit, fit["c"] * n_fit**fit["alpha"], color="C2",
            label=rf"Weighted fit to the real error: $\propto N^{{{exponent}}}$")

def plot_errors(ax, results, analysis):
    plot_error_points(ax, results, analysis["table"])
    plot_free_fit(ax, analysis["table"], analysis["fit"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Real and experimental error vs N")
    ax.set_xlabel("Needles thrown, N")
    ax.set_ylabel(r"Error in $\pi$")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

def loop_label(model):
    if model["beta"] is None:
        return "Needle loop (measured)"
    exponent = format_measurement(model["beta"], model["beta_uncertainty"], tex=True)
    return rf"Needle loop (measured), $\propto N^{{{exponent}}}$"

def model_label(model):
    per_needle = rf"{format_measurement(model['b'], model['b_uncertainty'], tex=True)}\,\mathrm{{s}} \times N"
    if model["a"] < 0.0005:
        return rf"Model: $T = {per_needle}$"
    serial = format_measurement(model["a"], model["a_uncertainty"], tex=True)
    return rf"Model: $T = ({serial})\,\mathrm{{s}} + {per_needle}$"

def plot_run_times(ax, times, model, config):
    n = times.index.to_numpy(dtype=float)
    ax.errorbar(n, times["time"], yerr=times["time_uncertainty"], fmt="o", capsize=2, label="Whole run (measured)")
    ax.errorbar(n, times["loop_time"], yerr=times["loop_time_uncertainty"], fmt="x", capsize=2, label=loop_label(model))
    n_line = np.logspace(np.log10(n[0]), np.log10(n[-1]), 200)
    ax.plot(n_line, predicted_seconds(model, n_line), color="C2", label=model_label(model))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"Run time vs N ({device_scope(config)})")
    ax.set_xlabel("Needles thrown, N")
    ax.set_ylabel("Time (s)")
    ax.legend(loc="center left", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

def time_label(seconds):
    years = seconds / SECONDS_PER_YEAR
    if years < 1e4:
        return human_time(seconds)
    return rf"${tex_number(years)}$ years"

def plot_prediction(ax, points, limit, label, color, linestyle="-", annotate=False, marker="o", show_beyond=True):
    decimals = np.array([point["decimals"] for point in points])
    seconds = np.array([point["seconds"] for point in points])
    reachable = decimals <= limit
    ax.plot(decimals[reachable], seconds[reachable], marker=marker, linestyle=linestyle, color=color, label=label)
    beyond = decimals >= limit
    if show_beyond and (~reachable).any():
        ax.plot(decimals[beyond], seconds[beyond], marker=marker, linestyle=":", color=color, alpha=0.25)
    if annotate:
        for d, s in zip(decimals[reachable], seconds[reachable]):
            ax.annotate(time_label(s), (d, s), textcoords="offset points", xytext=(-12, 9), fontsize=8)

def plot_measured_stars(ax, rows, label, filled):
    if not rows:
        return
    decimals = [-np.log10(row["error_measured"]) for row in rows]
    seconds = [row["time_measured"] for row in rows]
    if filled:
        ax.scatter(decimals, seconds, marker="*", s=140, zorder=3, color="C3", label=label)
    else:
        ax.scatter(decimals, seconds, marker="*", s=140, zorder=3, facecolors="none", edgecolors="C3", label=label)

def plot_measured(ax, check):
    used = [row for row in check["rows"] if row["in_fit"]]
    held_out = [row for row in check["rows"] if not row["in_fit"]]
    plot_measured_stars(ax, used, "Measured, used in the small-N fit", filled=False)
    plot_measured_stars(ax, held_out, "Measured, not used in that fit (the test)", filled=True)

def plot_reference_times(ax):
    for seconds, name in REFERENCE_TIMES:
        ax.axhline(seconds, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.text(DECIMALS[-1] + 0.6, seconds, name, color="gray", fontsize=8, verticalalignment="bottom")

def plot_64_bit_limit(ax, points):
    too_big = [point["decimals"] for point in points if point["n"] > LARGEST_N]
    if too_big:
        ax.axvline(too_big[0] - 0.5, color="gray", linestyle=":")
        ax.text(too_big[0] - 0.4, 0.02, r"Right of this line, $N > 9.2 \times 10^{18}$:" + "\nmore needles than one\n64-bit counter can hold",
                transform=ax.get_xaxis_transform(), fontsize=8, color="gray")

def add_needles_axis(ax):
    labels = []
    for decimals in DECIMALS:
        labels.append(f"${tex_number((K_THEORY / 10.0 ** -decimals) ** 2)}$")
    top = ax.twiny()
    top.set_xlim(ax.get_xlim())
    top.set_xticks(DECIMALS, labels, fontsize=7)
    top.set_xlabel(rf"Needles needed for d decimals (the same on every device): $N = ({K_THEORY:.2f} \times 10^d)^2$")

def shade_measured(ax, left, measured_decimals):
    ax.axvspan(left, measured_decimals, color="C2", alpha=0.1,
               label="Shaded: decimals actually measured in the runs; the rest is extrapolated")

def decorate_prediction(ax, points, some_unreachable, measured_decimals, left=DECIMALS[0] - 0.7):
    shade_measured(ax, left, measured_decimals)
    plot_reference_times(ax)
    plot_64_bit_limit(ax, points)
    if some_unreachable:
        ax.plot([], [], "o:", color="gray", alpha=0.4, label="Dotted: beyond what the precision can reach")
    ax.set_yscale("log")
    ax.set_xticks(DECIMALS)
    ax.set_xlim(left, DECIMALS[-1] + 3)
    ax.set_xlabel(r"Correct decimals of $\pi$ (error below $10^{-d}$)")
    ax.set_ylabel("Predicted time (s)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    add_needles_axis(ax)

def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def draw_error_figure(analysis, results, config, folder):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    plot_errors(axes[0], results, analysis)
    plot_run_times(axes[1], analysis["times"], analysis["model"], config)
    fig.suptitle(run_label(config))
    save_figure(fig, folder / "error.png")

def leftmost_measured_decimals(check):
    smallest = DECIMALS[0] - 0.7
    for row in check["rows"]:
        smallest = min(smallest, -np.log10(row["error_measured"]) - 0.3)
    return smallest

def draw_prediction_figure(analysis, config, folder):
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_prediction(ax, analysis["points"], analysis["limit"], "Predicted from all N", "C0", annotate=True)
    check = analysis["check"]
    if check:
        plot_prediction(ax, checkpoints(check["k"], check["k_uncertainty"], check), analysis["limit"],
                        rf"Predicted from $N \leq {tex_number(check['split'])}$ only", "C1", linestyle="--")
        plot_measured(ax, check)
    left = leftmost_measured_decimals(check) if check else DECIMALS[0] - 0.7
    decorate_prediction(ax, analysis["points"], analysis["limit"] < TARGET_DECIMALS, analysis["measured_decimals"], left)
    ax.set_title(rf"Predicted time for each decimal of $\pi$, from the measured speed ({device_scope(config)})", pad=35)
    fig.suptitle(run_label(config))
    save_figure(fig, folder / "prediction.png")

def statistics_summary(analysis):
    table = analysis["table"]
    fit = analysis["fit"]
    return {
        "largest_n": int(table.index[-1]),
        "k": fit["k"],
        "k_uncertainty": fit["k_uncertainty"],
        "k_bootstrap_uncertainty": fit["k_bootstrap_uncertainty"],
        "runs_used": fit["runs_used"],
        "k_theory": float(K_THEORY),
        "rmse_exponent": fit.get("alpha"),
        "rmse_exponent_uncertainty": fit.get("alpha_uncertainty"),
        "rmse_chi2": fit.get("chi2"),
        "rmse_dof": fit.get("dof"),
        "target_n_free_fit": free_fit_target(fit),
        "error_floor": analysis["floor"],
        "bias_at_largest_n": analysis["bias"],
        "bias_uncertainty": analysis["bias_uncertainty"],
        "sigma_at_largest_n": float(table["sigma"].iloc[-1]),
        "decimals_reached": analysis["reached"],
        "measured_decimals": analysis["measured_decimals"],
        "decimals_limit": analysis["limit"],
        "combined_pi": analysis["combined"],
    }

def time_summary(analysis):
    model = analysis["model"]
    largest_n = float(analysis["table"].index[-1])
    return {
        "serial_seconds": model["a"],
        "serial_seconds_uncertainty": model["a_uncertainty"],
        "seconds_per_needle": model["b"],
        "seconds_per_needle_uncertainty": model["b_uncertainty"],
        "loop_exponent": model["beta"],
        "loop_exponent_uncertainty": model["beta_uncertainty"],
        "needles_per_second": 1 / model["b"],
        "needles_per_second_uncertainty": model["b_uncertainty"] / model["b"] ** 2,
        "needles_per_second_total_time": float(largest_n / analysis["times"]["time"].iloc[-1]),
        "clock_spread": analysis["clock_spread"],
        "checkpoints": analysis["points"],
        "prediction_check": analysis["check"],
    }

def error_summary(analysis):
    summary = statistics_summary(analysis)
    summary.update(time_summary(analysis))
    return summary

def analyse_error(results, config, folder):
    analysis = analyse_errors(results, config)
    draw_error_figure(analysis, results, config, folder)
    draw_prediction_figure(analysis, config, folder)
    return error_summary(analysis)


# ============================================================ EP3: Amdahl's law

def amdahl(n, p):
    return 1 / ((1 - p) + p / n)

def fit_amdahl_p(sms, speedup, speedup_uncertainty):
    use = sms > 1
    x = 1 - 1 / sms[use]
    y = 1 - 1 / speedup[use]
    y_uncertainty = np.maximum(speedup_uncertainty[use] / speedup[use] ** 2, 1e-9)
    weight = 1 / y_uncertainty ** 2
    sxx = np.sum(weight * x * x)
    p = float(np.sum(weight * x * y) / sxx)
    chi2 = float(np.sum(weight * (y - p * x) ** 2))
    dof = int(use.sum()) - 1
    return {"p": p, "p_uncertainty": float(np.sqrt(1 / sxx)) * chi2_scale(chi2, dof), "chi2": chi2, "dof": dof}

def group_statistics(results):
    grouped = results.groupby(["n", "sms"])
    means = grouped.mean(numeric_only=True)
    uncertainties = grouped.std(ddof=1, numeric_only=True).div(np.sqrt(grouped.size()), axis=0).fillna(0)
    return means, uncertainties

def speedups_with_uncertainties(times, time_uncertainties):
    speedup = times[0] / times
    speedup_uncertainty = speedup * np.sqrt((time_uncertainties[0] / times[0]) ** 2 + (time_uncertainties / times) ** 2)
    speedup_uncertainty[0] = 0.0
    return speedup, speedup_uncertainty

def amdahl_fits(means, uncertainties, counts):
    sms = np.array(counts, dtype=float)
    fits = {}
    for label, column in SPEEDUPS:
        times = means.loc[counts, column].to_numpy(dtype=float)
        time_uncertainties = uncertainties.loc[counts, column].to_numpy(dtype=float)
        speedup, speedup_uncertainty = speedups_with_uncertainties(times, time_uncertainties)
        fit = fit_amdahl_p(sms, speedup, speedup_uncertainty)
        fit["speedup"] = speedup
        fit["speedup_uncertainty"] = speedup_uncertainty
        fits[label] = fit
    return fits

def bootstrap_p_uncertainty(results, n, counts, column, speedup_uncertainty):
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    sms = np.array(counts, dtype=float)
    samples = []
    for count in counts:
        samples.append(results[(results["n"] == n) & (results["sms"] == count)][column].to_numpy())
    p_values = []
    for _ in range(BOOTSTRAP_SAMPLES):
        times = np.array([generator.choice(values, len(values)).mean() for values in samples])
        p_values.append(fit_amdahl_p(sms, times[0] / times, speedup_uncertainty)["p"])
    return float(np.std(p_values, ddof=1))

def long_loop_clocks(results):
    long_loops = results[results["loop_time"] > TIMEABLE_LOOP_SECONDS]
    if len(long_loops) == 0:
        return results["measured_mhz"]
    return long_loops["measured_mhz"]

def speedup_ceiling(p, p_uncertainty):
    if p >= 1:
        return None, None
    return 1 / (1 - p), p_uncertainty / (1 - p) ** 2

def running_clock_mhz(results):
    timeable = results[results["loop_time"] > TIMEABLE_LOOP_SECONDS]
    if len(timeable) == 0:
        return float(results["clock_mhz"].iloc[0])
    return float(timeable["measured_mhz"].median())

def add_fixed_clock_times(results):
    results = results.copy()
    results["measured_mhz"] = results["cycles"] / results["loop_time"] / 1e6
    reference_mhz = running_clock_mhz(results)
    results["loop_time_fixed"] = results["cycles"] / (reference_mhz * 1e6)
    results["serial_time"] = results["time"] - results["loop_time"]
    results["time_fixed"] = results["serial_time"] + results["loop_time_fixed"]
    return results, reference_mhz

def amdahl_by_n(results, counts):
    means, uncertainties = group_statistics(results)
    by_n = {}
    for n in sorted(results["n"].unique()):
        n_means = means.loc[n]
        n_uncertainties = uncertainties.loc[n]
        if sorted(n_means.index) == counts:
            by_n[n] = (n_means.loc[counts], n_uncertainties.loc[counts], amdahl_fits(n_means, n_uncertainties, counts))
    return by_n

def plot_reference_amdahl_curves(ax, n_fit):
    for index, p_reference in enumerate((0.5, 0.9, 0.95, 0.99)):
        label = "Amdahl's law for other values of p" if index == 0 else None
        ax.plot(n_fit, amdahl(n_fit, p_reference), color="lightgray", linewidth=1, label=label)
        ax.text(n_fit[-1], amdahl(n_fit[-1], p_reference), f" p = {p_reference}", color="gray",
                verticalalignment="center")

def plot_speedup(ax, sms, fits, n_needles):
    n_fit = np.linspace(1, sms.max(), 200)
    ax.plot(n_fit, n_fit, linestyle=":", color="gray", label="Ideal: S(n) = n")
    plot_reference_amdahl_curves(ax, n_fit)
    for (label, column), color, marker in zip((SPEEDUPS[0], SPEEDUPS[2]), ("C0", "C2"), ("o", "x")):
        fit = fits[label]
        p = format_measurement(fit["p"], fit["p_uncertainty"], tex=True)
        ax.plot(n_fit, amdahl(n_fit, fit["p"]), color=color, linewidth=1.2)
        ax.errorbar(sms, fit["speedup"], yerr=fit["speedup_uncertainty"], fmt=marker, color=color, capsize=2, zorder=3,
                    label=rf"{SPEEDUP_NAMES[label]}: Amdahl fit $p = {p}$")
    ax.set_xticks(sms)
    ax.set_title(rf"Speedup vs number of SMs, $N = {tex_number(n_needles)}$")
    ax.set_xlabel("Number of SMs used, n")
    ax.set_ylabel(r"Speedup $S(n) = T(1)\,/\,T(n)$")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.4)

def plot_p_vs_n(ax, by_n):
    n = np.array(list(by_n), dtype=float)
    for label, _ in (SPEEDUPS[0], SPEEDUPS[2]):
        p_values = []
        p_uncertainties = []
        for _, _, fits in by_n.values():
            p_values.append(fits[label]["p"])
            p_uncertainties.append(fits[label]["p_uncertainty"])
        ax.errorbar(n, p_values, yerr=p_uncertainties, marker="o", capsize=2, label=SPEEDUP_NAMES[label])
    ax.axhline(1, color="gray", linestyle=":", linewidth=1)
    ax.set_xscale("log")
    ax.set_ylim(-0.15, 1.05)
    ax.set_title("Parallel fraction p vs N")
    ax.set_xlabel("Needles thrown, N")
    ax.set_ylabel(r"Parallel fraction $p$ (Amdahl fit, $\pm 1\sigma$)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.4)

def draw_amdahl_figure(results, by_n, n_main, counts, config, folder):
    sms = np.array(counts, dtype=float)
    fits = by_n[n_main][2]
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    plot_speedup(axes[0], sms, fits, n_main)
    plot_p_vs_n(axes[1], by_n)
    fig.suptitle(run_label(config))
    save_figure(fig, folder / "amdahl.png")

def amdahl_table(means, uncertainties, fits, counts, p):
    fit = fits[SPEEDUPS[0][0]]
    rows = []
    for i, count in enumerate(counts):
        rows.append({"sms": int(count), "time_fixed": float(means.loc[count, "time_fixed"]),
                     "time_fixed_uncertainty": float(uncertainties.loc[count, "time_fixed"]),
                     "time": float(means.loc[count, "time"]), "serial": float(means.loc[count, "serial_time"]),
                     "speedup": float(fit["speedup"][i]), "speedup_uncertainty": float(fit["speedup_uncertainty"][i]),
                     "amdahl": float(amdahl(count, p))})
    return rows

def p_by_n(by_n):
    values = {}
    for n, (_, _, fits) in by_n.items():
        fit = fits[SPEEDUPS[0][0]]
        values[int(n)] = (fit["p"], fit["p_uncertainty"])
    return values

def p_summary(results, fits, n_main, counts):
    whole = fits[SPEEDUPS[0][0]]
    wall = fits[SPEEDUPS[1][0]]
    loop = fits[SPEEDUPS[2][0]]
    return {
        "amdahl_n": int(n_main),
        "repetitions": int(len(results[(results["n"] == n_main) & (results["sms"] == 1)])),
        "p": whole["p"],
        "p_uncertainty": whole["p_uncertainty"],
        "p_chi2": whole["chi2"],
        "p_dof": whole["dof"],
        "p_bootstrap_uncertainty": bootstrap_p_uncertainty(results, n_main, counts, "time_fixed",
                                                           whole["speedup_uncertainty"]),
        "p_wall_clock": wall["p"],
        "p_wall_clock_uncertainty": wall["p_uncertainty"],
        "p_needle_loop": loop["p"],
        "p_needle_loop_uncertainty": loop["p_uncertainty"],
    }

def clock_summary(results, reference_mhz):
    clocks = long_loop_clocks(results)
    return {
        "measured_mhz": reference_mhz,
        "reported_mhz": int(results["clock_mhz"].iloc[0]),
        "clock_min_mhz": float(clocks.min()),
        "clock_max_mhz": float(clocks.max()),
    }

def amdahl_summary(results, by_n, n_main, counts, reference_mhz):
    means, uncertainties, fits = by_n[n_main]
    summary = p_summary(results, fits, n_main, counts)
    summary.update(clock_summary(results, reference_mhz))
    summary["p_by_n"] = p_by_n(by_n)
    summary["amdahl_table"] = amdahl_table(means, uncertainties, fits, counts, summary["p"])
    return summary

def analyse_amdahl(results, config, folder):
    results, reference_mhz = add_fixed_clock_times(results)
    counts = config["sm_counts"]
    by_n = amdahl_by_n(results, counts)
    if not by_n:
        return {}
    n_main = max(by_n)
    draw_amdahl_figure(results, by_n, n_main, counts, config, folder)
    return amdahl_summary(results, by_n, n_main, counts, reference_mhz)


# ============================================================ summary

def run_label(config):
    return f"{config['device'].upper()} FP{config['precision']}: {config['hardware']}"

def short_label(config):
    return f"{config['device'].upper()} FP{config['precision']}"

def device_name(config):
    if is_cpu(config):
        return "this CPU (one core)"
    return "this GPU (all SMs)"

def reachable(summary):
    return summary["decimals_limit"] >= TARGET_DECIMALS

def free_fit_line(summary):
    exponent = format_measurement(summary["rmse_exponent"], summary["rmse_exponent_uncertainty"])
    return (f"Error vs N, weighted fit: RMSE ~ N^({exponent}) (expected -0.5), "
            f"chi2/dof = {summary['rmse_chi2']:.1f}/{summary['rmse_dof']}")

def k_line(summary):
    k = format_measurement(summary["k"], summary["k_uncertainty"])
    k_pull = abs(summary["k"] - summary["k_theory"]) / summary["k_uncertainty"]
    return (f"Error vs N, slope fixed at -1/2: error = k/sqrt(N), k = {k} from {summary['runs_used']} runs "
            f"(bootstrap: ± {summary['k_bootstrap_uncertainty']:.2f}); theory {summary['k_theory']:.3f}, {k_pull:.1f} sigma away")

def combined_pi_line(summary):
    combined = summary["combined_pi"]
    pull = abs(combined["pi"] - np.pi) / combined["pi_uncertainty"]
    return (f"All {combined['needles']:.2e} needles of this run together: "
            f"pi = {format_measurement(combined['pi'], combined['pi_uncertainty'])}; the true pi is {pull:.1f} sigma away")

def floor_line(config, summary):
    bias = format_measurement(summary["bias_at_largest_n"], summary["bias_uncertainty"])
    return (f"The error stopped falling: at N = {summary['largest_n']:.1e} the estimate is off by {bias}, "
            f"far more than sigma = {summary['sigma_at_largest_n']:.1e}. Rounding in FP{config['precision']} "
            "biases each needle's test (so the combined pi above is biased too).")

def linearity_note(summary):
    beta = summary["loop_exponent"]
    pull = abs(beta - 1) / summary["loop_exponent_uncertainty"]
    if pull <= 2:
        return f", consistent with O(N) ({pull:.1f} sigma from 1)"
    if abs(beta - 1) < LINEAR_TOLERANCE:
        return f", close to O(N) but {pull:.1f} sigma from 1"
    return f", {pull:.1f} sigma from 1: not O(N)"

def time_model_line(summary):
    serial = format_measurement(summary["serial_seconds"], summary["serial_seconds_uncertainty"])
    per_needle = format_measurement(summary["seconds_per_needle"], summary["seconds_per_needle_uncertainty"])
    line = f"Time on the whole device: T = a + b N, a = {serial} s (serial part), b = {per_needle} s/needle"
    if summary["loop_exponent"] is None:
        return line
    exponent = format_measurement(summary["loop_exponent"], summary["loop_exponent_uncertainty"])
    return line + f"; the needle loop grows as N^({exponent}){linearity_note(summary)}"

def systematic_line(summary):
    return (f"Systematic: on the whole GPU the clock varied by {100 * summary['clock_spread']:.1f}% between long loops "
            "(laptop boost and heat); speeds and predicted times carry about the same relative uncertainty")

def fit_lines(config, summary):
    lines = []
    if summary.get("rmse_exponent") is not None:
        lines.append(free_fit_line(summary))
    lines.append(k_line(summary))
    lines.append(combined_pi_line(summary))
    if summary["error_floor"]:
        lines.append(floor_line(config, summary))
    lines.append(time_model_line(summary))
    if summary["clock_spread"] is not None:
        lines.append(systematic_line(summary))
    return lines

def ceiling_text(p, p_uncertainty):
    value, error = speedup_ceiling(p, p_uncertainty)
    if value is None or p + p_uncertainty >= 1:
        lower = 1 / (1 - (p - p_uncertainty)) if p - p_uncertainty < 1 else float("inf")
        return f"p is within 1 sigma of 1, so the speedup ceiling 1/(1-p) is only bounded below: > {lower:.0f}"
    return f"the speedup can never pass 1/(1-p) = {format_measurement(value, error)}"

def amdahl_table_lines(summary):
    lines = ["", f"Amdahl's law at N = {summary['amdahl_n']:.1e} (fixed clock = measured running clock; "
                 f"± is 1 sigma from {summary['repetitions']} repetitions per SM count):",
             f"{'SMs':>5} {'T fixed (s)':>18} {'T wall (s)':>11} {'serial (s)':>11} {'speedup':>16} {'Amdahl':>7}"]
    for row in summary["amdahl_table"]:
        lines.append(f"{row['sms']:>5} {format_measurement(row['time_fixed'], row['time_fixed_uncertainty']):>18} "
                     f"{row['time']:>11.3f} {row['serial']:>11.3f} "
                     f"{format_measurement(row['speedup'], row['speedup_uncertainty']):>16} {row['amdahl']:>7.2f}")
    return lines

def p_lines(summary):
    return [f"Fitted p = {format_measurement(summary['p'], summary['p_uncertainty'])} "
            f"(weighted, chi2/dof = {summary['p_chi2']:.1f}/{summary['p_dof']}; "
            f"bootstrap over the repetitions: ± {summary['p_bootstrap_uncertainty']:.1e})",
            f"Wall clock p = {format_measurement(summary['p_wall_clock'], summary['p_wall_clock_uncertainty'])}, "
            f"needle loop only p = {format_measurement(summary['p_needle_loop'], summary['p_needle_loop_uncertainty'])}",
            ceiling_text(summary["p"], summary["p_uncertainty"]),
            "The speedups share T(1), so they are correlated; the bootstrap includes that, the weighted fit does not."]

def p_by_n_line(summary):
    p_values = []
    for n, (p, p_uncertainty) in summary["p_by_n"].items():
        p_values.append(f"{float(n):.0e}: {format_measurement(p, p_uncertainty)}")
    return "p by N: " + ", ".join(p_values)

def clock_line(summary):
    return (f"GPU clock: median {summary['measured_mhz']:.0f} MHz while running (used as the fixed clock), "
            f"range {summary['clock_min_mhz']:.0f}-{summary['clock_max_mhz']:.0f} MHz over the long loops; "
            f"CUDA reports {summary['reported_mhz']} MHz. Runs near {summary['reported_mhz']} MHz did not boost; "
            "counting cycles removes that from the speedups.")

def amdahl_lines(summary):
    if "p" not in summary:
        return []
    return amdahl_table_lines(summary) + p_lines(summary) + [p_by_n_line(summary), clock_line(summary)]

def never_line(config, summary):
    precision = config["precision"]
    if summary["error_floor"]:
        why = "the error stopped falling (see above)."
    else:
        why = (f"FP{precision} keeps about {SIGNIFICANT_DIGITS[precision]} significant digits, "
               "so its needle test can never support 14 decimals.")
    return f"  NEVER with FP{precision}: {why} The numbers below are what it would take if the error kept falling."

def needles_line(summary, target):
    line = f"  Needles needed: {format_measurement(target['n'], target['n_uncertainty'])} = (k / 1e-14)^2"
    if summary.get("target_n_free_fit"):
        decades = np.log10(target["n"] / summary["largest_n"])
        line += (f" (the free fit would say {summary['target_n_free_fit']:.1e}: its slope is off by "
                 f"{abs(summary['rmse_exponent'] + 0.5):.3f}, stretched over {decades:.0f} powers of ten)")
    return line

def speed_line(config, summary):
    speed = format_measurement(summary["needles_per_second"], summary["needles_per_second_uncertainty"])
    line = f"  Speed of {device_name(config)}: {speed} needles/s (statistical)"
    if summary["clock_spread"] is not None:
        line += f" ± {100 * summary['clock_spread']:.1f}% (systematic, clock)"
    if not is_cpu(config):
        line += (f"; {summary['needles_per_second_total_time']:.2e} counting the serial part, "
                 f"at N = {summary['largest_n']:.1e}")
    return line

def lifetime_lines(config, years):
    if is_cpu(config):
        return [f"  In {LIFETIME_YEARS} years you would need {years / LIFETIME_YEARS:.1e} cores like this one"]
    sms = config["sm_counts"][-1]
    return [f"  For N this large the serial part is nothing, so p -> 1 and time falls as 1/n (Amdahl):",
            f"  in {LIFETIME_YEARS} years you would need {years / LIFETIME_YEARS:.1e} GPUs like this one "
            f"({years / LIFETIME_YEARS * sms:.1e} SMs)"]

def target_lines(config, summary):
    target = summary["checkpoints"][-1]
    years = target["seconds"] / SECONDS_PER_YEAR
    relative = target["seconds_uncertainty"] / target["seconds"]
    lines = ["", f"EP1 question 3: pi with 14 decimals (error {TARGET_ERROR:.0e})"]
    if not reachable(summary):
        lines.append(never_line(config, summary))
    lines.append(needles_line(summary, target))
    lines.append(speed_line(config, summary))
    lines.append(f"  Time: {format_measurement(target['seconds'], target['seconds_uncertainty'])} s = "
                 f"{human_time(target['seconds'])} ± {100 * relative:.0f}%")
    lines += lifetime_lines(config, years)
    lines.append(f"  N = {target['n']:.1e} does not fit in a 64-bit integer (max 9.2e18): it can only be extrapolated.")
    return lines

def check_ratios(row):
    error_ratio = ratio_with_uncertainty(row["error_measured"], row["error_measured_uncertainty"],
                                         row["error_predicted"], row["error_predicted_uncertainty"])
    time_ratio = ratio_with_uncertainty(row["time_measured"], max(row["time_measured_uncertainty"], TIME_RESOLUTION),
                                        row["time_predicted"], row["time_predicted_uncertainty"])
    return error_ratio, time_ratio

def check_row_line(row, error_ratio, time_ratio):
    return (f"  {row['n']:>8.1e} {row['error_predicted']:>12.2e} {row['error_measured']:>12.2e} "
            f"{format_measurement(*error_ratio):>14} {row['time_predicted']:>10.3f}s "
            f"{row['time_measured']:>10.3f}s {format_measurement(*time_ratio):>14}")

def check_notes(summary):
    notes = []
    if summary["clock_spread"] is not None:
        notes.append(f"  The time ratios are statistical only; the clock systematic ({100 * summary['clock_spread']:.1f}%) "
                     "also applies to them.")
    if summary["error_floor"]:
        notes.append("  The error check fails here on purpose: the small-N fit expects the error to keep falling, "
                     "but rounding stops it (the floor above).")
    return notes

def check_lines(summary):
    check = summary.get("prediction_check")
    if not check:
        return []
    lines = ["", f"Prediction check: fitted on N <= {check['split']:.0e} only, then compared with the larger N measured",
             f"  {'N':>8} {'error pred.':>12} {'error meas.':>12} {'ratio':>14} {'time pred.':>11} {'time meas.':>11} {'ratio':>14}"]
    error_ratios = []
    time_ratios = []
    for row in check["rows"]:
        if row["in_fit"]:
            continue
        error_ratio, time_ratio = check_ratios(row)
        error_ratios.append(error_ratio)
        time_ratios.append(time_ratio)
        lines.append(check_row_line(row, error_ratio, time_ratio))
    lines.append(f"  Weighted mean of measured/predicted: {format_measurement(*weighted_mean(error_ratios))} for the error, "
                 f"{format_measurement(*weighted_mean(time_ratios))} for the time (1 means the prediction is right)")
    return lines + check_notes(summary)

def checkpoint_note(config, summary, point):
    if point["decimals"] <= summary["decimals_reached"]:
        return "measured in this run"
    if point["decimals"] > summary["decimals_limit"]:
        if summary["error_floor"]:
            return "never: beyond the error floor"
        return f"never: beyond FP{config['precision']}"
    if point["n"] > LARGEST_N:
        return "N > 9.2e18: more than one run"
    return ""

def checkpoint_lines(config, summary):
    lines = ["", f"Checkpoints on {device_name(config)} (error 10^-d = k/sqrt(N), time = a + bN, ± 1 sigma statistical):",
             f"  {'decimals':>8} {'needles':>22} {'time':>16} {'±':>6}"]
    for point in summary["checkpoints"]:
        relative = point["seconds_uncertainty"] / point["seconds"]
        lines.append(f"  {point['decimals']:>8} {format_measurement(point['n'], point['n_uncertainty']):>22} "
                     f"{human_time(point['seconds']):>16} {100 * relative:>5.0f}%  {checkpoint_note(config, summary, point)}")
    return lines

def summary_text(config, summary):
    title = run_label(config)
    lines = [title, "=" * len(title)]
    lines += fit_lines(config, summary)
    lines += amdahl_lines(summary)
    lines += target_lines(config, summary)
    lines += check_lines(summary)
    lines += checkpoint_lines(config, summary)
    return "\n".join(lines)

def analyse_run(folder):
    config, results = load_run(folder)
    summary = analyse_error(results, config, folder)
    if not is_cpu(config):
        summary.update(analyse_amdahl(results, config, folder))
    text = summary_text(config, summary)
    (folder / "summary.txt").write_text(text + "\n")
    print("\n" + text)
    return summary


# ============================================================ comparing runs

def load_runs(folders):
    runs = []
    for folder in folders:
        config, results = load_run(folder)
        runs.append((config, results, analyse_run(folder)))
    return runs

PRECISION_STYLES = {
    64: {"marker": "o", "linestyle": "-", "markersize": 7, "markerfacecolor": None},
    32: {"marker": "s", "linestyle": "--", "markersize": 7, "markerfacecolor": "none"},
    16: {"marker": "^", "linestyle": "-", "markersize": 6, "markerfacecolor": None},
}

def precision_style(precision, color):
    style = dict(PRECISION_STYLES[precision])
    if style["markerfacecolor"] is None:
        style["markerfacecolor"] = color
    return style

def plot_report_errors(ax, runs):
    largest = 0
    for index, (config, results, _) in enumerate(runs):
        table = error_table(results)
        color = f"C{index}"
        ax.errorbar(table.index, table["rmse"], yerr=table["rmse_uncertainty"], color=color, capsize=2,
                    label=short_label(config), **precision_style(config["precision"], color))
        largest = max(largest, table.index.max())
    n = np.logspace(np.log10(FIRST_N), np.log10(largest), 100)
    ax.plot(n, K_THEORY / np.sqrt(n), "k--", linewidth=1, label=rf"Theory: ${K_THEORY:.2f}/\sqrt{{N}}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Real error vs N, all runs")
    ax.set_xlabel("Needles thrown, N")
    ax.set_ylabel(r"Real error in $\pi$ (root mean square over the runs, $\pm 1\sigma$)")
    ax.legend(fontsize=8, title="FP64 and FP32 use the same random numbers,\nso their curves overlap", title_fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

def times_faster(ratio):
    if ratio < 10:
        return f"{ratio:.1f}"
    return f"{ratio:.0f}"

def speed_label(config, summary, baseline, column):
    speed = format_measurement(summary["needles_per_second"], summary["needles_per_second_uncertainty"], tex=True)
    return (rf"{short_label(config)}: ${speed}$ needles/s, "
            f"{times_faster(summary['needles_per_second'] / baseline)}{column}")

def plot_report_predictions(ax, runs):
    baseline, column = speed_baseline(runs)
    some_unreachable = False
    for index, (config, _, summary) in enumerate(runs):
        plot_prediction(ax, summary["checkpoints"], summary["decimals_limit"],
                        speed_label(config, summary, baseline, column), f"C{index}",
                        linestyle=PRECISION_STYLES[config["precision"]]["linestyle"],
                        marker=PRECISION_STYLES[config["precision"]]["marker"], show_beyond=False)
        some_unreachable = some_unreachable or summary["decimals_limit"] < TARGET_DECIMALS
    if some_unreachable:
        ax.plot([], [], " ", label="Each line stops at the last decimal its precision can reach")
    measured = max(summary["measured_decimals"] for _, _, summary in runs)
    decorate_prediction(ax, runs[0][2]["checkpoints"], False, measured)
    ax.set_title(r"Predicted time for each decimal of $\pi$, from each run's measured speed", pad=35)

def draw_report_figure(runs, out):
    fig, (error_axis, prediction_axis) = plt.subplots(1, 2, figsize=(22, 8.5))
    plot_report_errors(error_axis, runs)
    plot_report_predictions(prediction_axis, runs)
    save_figure(fig, out / "report.png")

def speed_baseline(runs):
    for config, _, summary in runs:
        if is_cpu(config) and config["precision"] == 64:
            return summary["needles_per_second"], "× CPU FP64"
    slowest = min(summary["needles_per_second"] for _, _, summary in runs)
    return slowest, "× slowest"

def speed_table_lines(runs):
    baseline, column = speed_baseline(runs)
    lines = [f"{'Run':<50} {'needles/s':>22} {column:>10} {'14 decimals':>16} {'p':>20}"]
    for config, _, summary in runs:
        time_14 = human_time(summary["checkpoints"][-1]["seconds"]) if reachable(summary) else "never"
        p = format_measurement(summary["p"], summary["p_uncertainty"]) if "p" in summary else "-"
        speed = format_measurement(summary["needles_per_second"], summary["needles_per_second_uncertainty"])
        lines.append(f"{run_label(config)[:50]:<50} {speed:>22} "
                     f"{summary['needles_per_second'] / baseline:>10.1f} {time_14:>16} {p:>20}")
    return lines

def decimals_cell(summary, index, decimals):
    if decimals > summary["decimals_limit"]:
        return "never"
    return human_time(summary["checkpoints"][index]["seconds"])

def decimals_table_lines(runs):
    header = f"{'decimals':>8} "
    for config, _, _ in runs:
        header += f"{short_label(config):>16} "
    lines = ["", "Time for each number of correct decimals:", header]
    for index, decimals in enumerate(DECIMALS):
        line = f"{decimals:>8} "
        for _, _, summary in runs:
            line += f"{decimals_cell(summary, index, decimals):>16} "
        lines.append(line)
    return lines

def report(folders):
    runs = load_runs(folders)
    out = RUNS / f"{datetime.now():%Y-%m-%d_%H-%M-%S}-report"
    out.mkdir(parents=True)
    draw_report_figure(runs, out)
    text = "\n".join(speed_table_lines(runs) + decimals_table_lines(runs))
    (out / "report.txt").write_text(text + "\n")
    print("\n" + text)
    print(f"\nReport in {out}")


# ============================================================ GPU info

def show_hardware(info):
    print(f"\nGPU: {info['NAME'].replace('_', ' ')} (compute capability {info['COMPUTE_CAPABILITY']}), "
          f"{info['MEMORY_GB']} GB, {info['CLOCK_MHZ']} MHz")
    print("\nHardware (what actually computes)")
    print(f"  SMs (streaming multiprocessors):   {info['SMS']}")
    print(f"  CUDA cores (FP32 units) per SM:    {info['CUDA_CORES_PER_SM']}")
    print(f"  FP32 units per FP64 unit:          {info['FP32_PER_FP64']}")
    print(f"  Warp size:                         {info['WARP_SIZE']} threads run one instruction together")

def show_limits(info):
    threads_per_sm = int(info["MAX_THREADS_PER_SM"])
    warps = threads_per_sm // int(info["WARP_SIZE"])
    print("\nLimits (how much work an SM can hold at once)")
    print(f"  Threads per block, max:            {info['MAX_THREADS_PER_BLOCK']}")
    print(f"  Threads resident per SM, max:      {threads_per_sm} = {warps} warps")
    print(f"  Blocks resident per SM, max:       {info['MAX_BLOCKS_PER_SM']}")
    print(f"  Registers per SM:                  {info['REGISTERS_PER_SM']}")

def show_launch(info):
    sms = int(info["SMS"])
    threads = int(info["THREADS_PER_BLOCK"])
    blocks_per_sm = int(info["BLOCKS_PER_SM"])
    filled = blocks_per_sm * threads / int(info["MAX_THREADS_PER_SM"])
    cooperative = "yes" if info["COOPERATIVE_LAUNCH"] == "1" else "no"
    print("\nThis experiment")
    print(f"  The kernel uses {info['REGISTERS_PER_THREAD']} registers per thread. The occupancy API fills each SM with")
    print(f"  {blocks_per_sm} x {threads}-thread blocks = {blocks_per_sm * threads} threads ({filled:.0%} of the SM's thread limit).")
    print(f"  Every run launches {blocks_per_sm * sms} blocks (the whole GPU); blocks on SMs outside the n being")
    print(f"  tested end at once. SM ids found: {info['SM_IDS_FOUND']}. Cooperative launch: {cooperative}.")
    print(f"  SM counts in the Amdahl sweep: {sm_counts(sms)}")

def show_info(info):
    show_hardware(info)
    show_limits(info)
    show_launch(info)


# ============================================================ main

def default_repetitions(args):
    if args.repetitions:
        return args.repetitions
    return 20 if args.cpu else 3

def build_config(args, gpu):
    device = "cpu" if args.cpu else "gpu"
    return {
        "device": device,
        "precision": args.precision,
        "budget": args.budget,
        "n": args.n,
        "seed": args.seed,
        "repetitions": default_repetitions(args),
        "sm_counts": [1] if args.cpu else sm_counts(int(gpu["SMS"])),
        "hardware": read_cpu_name() if args.cpu else gpu["NAME"].replace("_", " "),
        "gpu": gpu,
    }

def has_results(folder):
    with open(folder / "results.csv") as file:
        return len(file.readlines()) > 1

def run(config):
    folder = new_run_folder(f"{config['device']}-fp{config['precision']}", config)
    print(f"{run_label(config)}. Results go to {folder}")
    started = time.perf_counter()
    run_experiment(config, folder)
    print(f"Total elapsed time: {time.perf_counter() - started:.0f} s")
    if not has_results(folder):
        print("No run finished, so there is nothing to analyse.")
        return
    analyse_run(folder)
    print(f"\nDone. Graphs and summary in {folder}")

def main():
    args = parse_args()
    if args.plot:
        analyse_run(args.plot)
        return
    if args.report:
        report(args.report)
        return
    compile_engine()
    if args.info:
        show_info(read_gpu_info())
        return
    gpu = {} if args.cpu else read_gpu_info()
    run(build_config(args, gpu))

if __name__ == "__main__":
    main()
