import argparse
import json
import os
import shutil
import time
from datetime import datetime

import engine
from gpu_monitor import GpuMonitor

RUNS_FOLDER = os.path.join(engine.HERE, "runs")
SM_COUNTS = [1, 2, 3, 4, 6, 9, 12, 18, 24, 36]
PROBLEM_SIZES = [10 ** 5, 10 ** 6, 10 ** 7, 10 ** 8, 10 ** 9, 10 ** 10]
CALIBRATION_PHOTONS = {64: 10 ** 8, 32: 10 ** 9}
DEVICES = [("cpu", 64), ("cpu", 32), ("gpu", 64), ("gpu", 32)]
ALL_SWEEPS = [("devices", None), ("problem-size", 32), ("weak", 64), ("weak", 32), ("strong", 64), ("strong", 32)]


# ============================================================ one point

def point_name(point):
    sms = f"_sms{point['sms']:02d}" if point["sms"] else ""
    return f"point_{point['processor']}{point['precision']}{sms}_n{point['photons']:.2e}_rep{point['repeat']}.json"


def save_json(data, path):
    partial_path = path + ".partial"
    with open(partial_path, "w") as file:
        json.dump(data, file, indent=2)
    os.replace(partial_path, path)


def read_json(path):
    with open(path) as file:
        return json.load(file)


def run_point(point, scratch_folder):
    config = engine.default_star_config(point["photons"], sms=point["sms"])
    shutil.rmtree(scratch_folder, ignore_errors=True)
    started_at = time.perf_counter()
    with GpuMonitor() as monitor:
        output = engine.run_engine_output(config, scratch_folder, point["processor"], point["precision"])
    process_time = time.perf_counter() - started_at
    shutil.rmtree(scratch_folder, ignore_errors=True)
    result = dict(point)
    result["SUMMARY"] = engine.lines_of_kind(output, "SUMMARY")[0]
    result["KERNEL"] = engine.lines_of_kind(output, "KERNEL")
    result["SMS"] = engine.lines_of_kind(output, "SMS")
    result["GPU_MONITOR"] = monitor.summary()
    result["PROCESS_TIME"] = process_time
    return result


# ============================================================ the sweeps' points

def speed_on_all_units(processor, precision):
    photons = CALIBRATION_PHOTONS[precision] if processor == "gpu" else 10 ** 7
    point = {"processor": processor, "precision": precision, "sms": None, "photons": photons, "repeat": 0}
    result = run_point(point, os.path.join(RUNS_FOLDER, "calibration-scratch"))
    return result["SUMMARY"]["PHOTONS_PER_SECOND"]


def base_points(kind, precision, seconds, photons=None):
    if kind == "strong" and photons:
        return [{"processor": "gpu", "precision": precision, "sms": sms, "photons": photons} for sms in SM_COUNTS]
    if kind == "devices":
        return [{"processor": processor, "precision": p, "sms": None, "photons": 10 ** 8}
                for processor, p in DEVICES]
    if kind == "problem-size":
        return [{"processor": "gpu", "precision": precision, "sms": sms, "photons": photons}
                for photons in PROBLEM_SIZES for sms in (1, SM_COUNTS[-1])]
    speed = speed_on_all_units("gpu", precision)
    if kind == "strong":
        photons = int(seconds * speed)
        return [{"processor": "gpu", "precision": precision, "sms": sms, "photons": photons} for sms in SM_COUNTS]
    photons_per_sm = int(seconds * speed / SM_COUNTS[-1])
    return [{"processor": "gpu", "precision": precision, "sms": sms, "photons": photons_per_sm * sms}
            for sms in SM_COUNTS]


def interleaved(points, repeats):
    return [dict(point, repeat=repeat) for repeat in range(1, repeats + 1) for point in points]


def describe_sweep(kind, precision, seconds, repeats, photons=None):
    return {"kind": kind, "precision": precision, "seconds_on_all_sms": seconds, "photons": photons,
            "repeats": repeats, "created": datetime.now().isoformat(timespec="seconds"),
            "code_hash": engine.code_hash(), "points": interleaved(base_points(kind, precision, seconds, photons),
                                                                  repeats)}


def new_sweep_folder(kind, precision, photons=None):
    label = kind + (f"-gpu{precision}" if precision else "") + (f"-n{photons:.0e}" if photons else "")
    folder = os.path.join(RUNS_FOLDER, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"-sweep-{label}")
    os.makedirs(folder)
    return folder


# ============================================================ running a sweep

def warm_up(points):
    gpu_points = [point for point in points if point["processor"] == "gpu"]
    if gpu_points:
        speed_on_all_units("gpu", gpu_points[0]["precision"])


def run_sweep(folder):
    sweep = read_json(os.path.join(folder, "sweep.json"))
    points = sweep["points"]
    missing = [point for point in points if not os.path.exists(os.path.join(folder, point_name(point)))]
    print(f"{folder}: {len(points) - len(missing)} of {len(points)} points done", flush=True)
    warm_up(missing)
    for number, point in enumerate(missing, 1):
        result = run_point(point, os.path.join(folder, "scratch"))
        save_json(result, os.path.join(folder, point_name(point)))
        print(f"  {number}/{len(missing)} {point_name(point)}: photons {result['SUMMARY']['PHOTONS_TIME']:.3f} s, "
              f"total {result['SUMMARY']['TIME']:.3f} s", flush=True)
    return folder


def start_sweep(kind, precision, seconds, repeats, photons=None):
    folder = new_sweep_folder(kind, precision, photons)
    save_json(describe_sweep(kind, precision, seconds, repeats, photons), os.path.join(folder, "sweep.json"))
    return run_sweep(folder)


def parse_args():
    parser = argparse.ArgumentParser(description="EP4's scaling sweeps: one engine process per point")
    parser.add_argument("kind", choices=["strong", "weak", "problem-size", "devices", "all", "continue"])
    parser.add_argument("folder", nargs="?", help="for continue: the sweep's folder")
    parser.add_argument("--precision", type=int, choices=engine.PRECISIONS, default=64)
    parser.add_argument("--seconds", type=float, default=5.0, help="the photon phase on all SMs")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--photons", type=float, default=None, help="strong: a fixed N instead of the calibrated one")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.kind == "continue":
        run_sweep(args.folder)
    elif args.kind == "all":
        for kind, precision in ALL_SWEEPS:
            start_sweep(kind, precision, args.seconds, args.repeats)
    else:
        photons = int(args.photons) if args.photons else None
        start_sweep(args.kind, None if args.kind == "devices" else args.precision, args.seconds, args.repeats, photons)


if __name__ == "__main__":
    main()
