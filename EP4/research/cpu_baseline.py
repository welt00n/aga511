import os

import common
from lab import engine
from lab.gpu_monitor import GpuMonitor

SHORT = 1e7
LONG = 1e8
GPU_WARM_UP = 3e11


def cpu_speed(photons, prefix=()):
    summary = common.engine_summary(engine.default_star_config(photons), "cpu", 64, prefix)
    return photons / summary["PHOTONS_TIME"]


def pinned_to_each_cpu():
    print(f"Pinned to each logical CPU (taskset -c n), N={SHORT:.0e}:")
    for cpu in range(os.cpu_count()):
        print(f"  cpu {cpu:2d}: {cpu_speed(SHORT, ['taskset', '-c', str(cpu)]):.2e} photons/s", flush=True)


def not_pinned(repeats=8):
    print(f"\nNot pinned, {repeats} runs, N={SHORT:.0e}:")
    print("  " + "  ".join(f"{cpu_speed(SHORT):.2e}" for _ in range(repeats)), flush=True)


def with_and_without_the_monitor(repeats=2):
    print(f"\nAlone vs with nvidia-smi sampling every 200 ms, N={LONG:.0e}:")
    for repeat in range(repeats):
        alone = cpu_speed(LONG)
        with GpuMonitor():
            monitored = cpu_speed(LONG)
        print(f"  repeat {repeat + 1}: alone {alone:.2e}, monitored {monitored:.2e}, ratio {monitored / alone:.3f}",
              flush=True)


def before_and_after_the_gpu():
    print(f"\nBefore and right after a long GPU FP32 run (N={GPU_WARM_UP:.0e}), N={LONG:.0e}:")
    before = cpu_speed(LONG)
    gpu = common.engine_summary(engine.default_star_config(GPU_WARM_UP), "gpu", 32)
    after = cpu_speed(LONG)
    print(f"  before {before:.2e}, GPU ran {gpu['PHOTONS_TIME']:.1f} s, after {after:.2e}, ratio {after / before:.3f}")


with common.recorded(__file__):
    print("Q4: how much does the one-core CPU baseline (FP64) move, and what moves it?\n")
    pinned_to_each_cpu()
    not_pinned()
    with_and_without_the_monitor()
    before_and_after_the_gpu()
    print("\nFound on 2026-11-22: 8.8-9.1e6 photons/s in every case, against 7.0-8.0e6 in Saturday's sweep.")
    print("Not the run length, the GPU's heat or nvidia-smi. A re-run the same afternoon gave 7.6-9.4e6 pinned")
    print("(by core) and 8.0-8.9e6 not pinned: the CPU's own spread (~20%) covers the sweep's gap.")
