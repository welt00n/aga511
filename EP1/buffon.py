import time
from datetime import datetime
import argparse
from tqdm import tqdm
import subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# LINK_TO_COLLAB="https://colab.research.google.com/drive/1k3lFfxoS_p-HhlmDtAQS_6qJdhhLcMSy#scrollTo=Vxug-cFT6ym5"
def init():
    global ARGS
    global ENGINES
    ARGS = parse_args()
    ENGINES = {
        "single-thread-c": run,
        "thread-pool-c": run_pool
    }

def parse_args():
    parser = argparse.ArgumentParser(description="Buffon's needle Monte Carlo Experiment")
    parser.add_argument(
        "--range",
        nargs=2,
        type=int,
        default=[6,10],
        metavar=("min", "max"),
        help="Minimum and maximum powers of 10 for N"
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=5,
        help="Number of independent experiments for each N"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global seed to generate the experimental seeds"
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="single-thread-c",
        help="Threadpool for faster experimentation, single-thread for single thread performance analysis",
        choices=("single-thread-c", "thread-pool-c")
    )
    return parser.parse_args()

def welcome_message():
    print(f"Starting Buffon's exepriment")

def run_c_engine(n, seed):
    result = subprocess.run(
        ["./buffon", str(n), str(seed)],
        capture_output=True,
        text=True,
    )
    if result.returncode!=0:
        raise RuntimeError(f"Error running C engine:\n{result.stderr}")
    return {item.split("=")[0]:item.split("=")[1] for item in result.stdout.split()}

def compile_c_engine():
    result = subprocess.run(
        ["gcc", "buffon.c", "-o", "buffon", "-O2", "-lm"],
        capture_output=True,
        text=True,
        check=False
    )
    if result.stderr:
        raise(Exception(f"Error compiling C engine:\n{result.stderr}"))

def run() -> pd.DataFrame:
    results = []
    compile_c_engine()
    for order in range(ARGS.range[0], ARGS.range[1]+1):
        N = 10**order
        for index, repetition in tqdm(enumerate(range(ARGS.repetitions))):
            seed = ARGS.seed+repetition+N
            result = run_c_engine(N, seed)
            results.append({
                "n":N,
                "pi": float(result.get('PI')),
                'error': float(result.get('ERROR')),
                'time': int(result.get('TIME'))
            })
    result = pd.DataFrame(results)
    return result

def run_pool() -> pd.DataFrame:
    compile_c_engine()
    results = []
    experiments = []
    for order in range(ARGS.range[0], ARGS.range[1]+1):
        N = 10**order
        for repetition in range(ARGS.repetitions):
            seed = ARGS.seed+repetition+N
            experiments.append({"n":N, "seed":seed})

    def run_experiment(experiment):
        result = run_c_engine(experiment['n'], experiment['seed'])
        return {
            "n":experiment['n'],
            "pi": float(result.get('PI')),
            'error': float(result.get('ERROR')),
            'time': int(result.get('TIME'))
        }

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(run_experiment, experiments), total=len(experiments), desc="Running experiments"))
    
    result = pd.DataFrame(results)
    return result

def finish_message():
    print("\nDone.")

def plot_pi_vs_n(ax, pi, n):
    ax.scatter(n, pi, label=r'Mean Estimated $\pi$')
    ax.set_xscale("log")
    ax.axhline(np.pi, linestyle='--', label='Reference pi')
    ax.set_title(r'Mean estimated $\pi$ vs $N$')
    ax.set_xlabel("N")
    ax.set_ylabel(r"Mean Estimated $\pi$")
    ax.set_ylim(3.141, 3.142)
    ax.legend() # test it
    ax.grid(True)

def plot_n_vs_rmse(ax, n_fit, rmse_fit, rmse_theory, target_error, alpha, r_squared):
    ax.plot(n_fit, rmse_fit, label=rf"RMSE fit $\propto N^{{{alpha:.3f}}}$")
    ax.plot(n_fit, rmse_theory, linestyle="--", label=r"RMSE theory $\propto N^{-1/2}$")
    ax.axhline( target_error, color="red", linestyle="--", label=rf"Target RMSE = {target_error:.2e}")
    ax.text(0.05, 0.8, rf"$\alpha = {alpha:.4f}$" + "\n" + rf"$R^2 = {r_squared:.4f}$", transform=ax.transAxes, verticalalignment="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("RMSE vs N")
    ax.set_xlabel("N")
    ax.set_ylabel("RMSE")
    ax.legend()
    ax.grid(True)


def plot_n_vs_time(ax,n,elapsed_time,n_fit,time_fit,target_n,target_time,beta,r_squared_time):
    ax.plot(n, elapsed_time, marker='o', label='Measured runtime')
    ax.plot(n_fit, time_fit, label=rf"Runtime fit $\propto N^{{{beta:.3f}}}$")
    ax.axvline(target_n, linestyle=":", label=rf"Target $N \approx {target_n:.2e}$")
    ax.axhline(target_time, linestyle="--", label=rf"Estimated time $\approx {target_time:.2e}$ s" )
    ax.text(0.05, 0.8, rf"$\beta = {beta:.4f}$" + "\n" + rf"$R^2 = {r_squared_time:.4f}$", transform=ax.transAxes, verticalalignment="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Average Runtime vs N")
    ax.set_xlabel("N")
    ax.set_ylabel("Average Runtime (s)")
    ax.legend()
    ax.grid(True)

def run_analysis(results):
    target_error = 1e-14
    aggregated = results.groupby("n").agg(
        average_time=("time", "mean"),
        pi=("pi", "mean"),
        rmse=("error", lambda x: np.sqrt(np.mean(x**2))),
    )

    n = aggregated.index.to_numpy(dtype=float)
    pi = aggregated["pi"].to_numpy(dtype=float)
    rmse = aggregated["rmse"].to_numpy(dtype=float)
    elapsed_time = aggregated["average_time"].to_numpy(dtype=float)

    log_n = np.log(n)
    log_rmse = np.log(rmse)

    alpha, log_k = np.polyfit(log_n, log_rmse, 1)
    k = np.exp(log_k)
    predicted_log_rmse = log_k + alpha * log_n

    res = np.sum((log_rmse - predicted_log_rmse) ** 2)
    tot = np.sum((log_rmse - np.mean(log_rmse)) ** 2)
    r_squared = 1 - res / tot

    p = 2 / np.pi
    k_theory = np.sqrt(4 * (1 - p) / p**3)
    target_n = (target_error / k) ** (1 / alpha)

    n_fit = np.logspace(
        np.log10(n.min()),
        np.log10(n.max()),
        200
    )
    rmse_fit = k * n_fit**alpha
    rmse_theory = k_theory / np.sqrt(n_fit)

    # Got a bug here, if the time elapsed is below 1 second we get an error (~N<10^4)
    valid = elapsed_time > 0
    valid_n = n[valid]
    valid_time = elapsed_time[valid]

    log_valid_n = np.log(valid_n)
    log_valid_time = np.log(valid_time)
    beta, log_c = np.polyfit(log_valid_n, log_valid_time, 1)
    predicted_log_time = log_c + beta * log_valid_n
    res_time = np.sum((log_valid_time - predicted_log_time) ** 2)
    tot_time = np.sum((log_valid_time - np.mean(log_valid_time)) ** 2)
    r_squared_time = 1 - res_time / tot_time

    c = np.exp(log_c)
    time_fit = c * n_fit**beta
    rate = valid_n / valid_time
    measured_rate = rate[-1]
    target_time = target_n / measured_rate

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    plot_pi_vs_n(axes[0], pi, n)
    plot_n_vs_rmse(
        axes[1],
        n_fit,
        rmse_fit,
        rmse_theory,
        target_error,
        alpha,
        r_squared
    )
    plot_n_vs_time(
        axes[2],
        n,
        elapsed_time,
        n_fit,
        time_fit,
        target_n,
        target_time,
        beta,
        r_squared_time
    )

    print("\nRMSE scaling")
    print("------------")
    print(f"Fit: RMSE = {k:.6e} * N^({alpha:.6f})")
    print(f"R²: {r_squared:.6f}")
    print(f"Theoretical exponent: -0.5")

    print("\nPrecision")
    print("-----------------------")
    print(f"Target RMSE: {target_error:.1e}")
    print(f"Estimated N: {target_n:.6e}")

    print("\nRuntime scaling")
    print("---------------")
    print(f"Fit: T = {c:.6e} * N^({beta:.6f})")
    print(f"R²: {r_squared_time:.6f}")
    print(f"Measured throughput: {measured_rate:.6e} throws/s")
    print(f"Estimated target runtime: {target_time:.6e} s")

    # --------------------------------------------------
    # Save figure
    # --------------------------------------------------

    plt.tight_layout()

    fig.savefig(
        datetime.now().strftime(
            "%d-%m-%Y-%H:%M:%S-results.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

if __name__ == '__main__':
    init()
    welcome_message()
    start = time.perf_counter()
    run_func = ENGINES[ARGS.engine]
    results = run_func()
    total_elapsed = time.perf_counter() - start
    print(f"Total elapsed time: {total_elapsed}\n Running analysis...", results)
    run_analysis(results)
    finish_message()