import glob
import hashlib
import os
import shutil
import time

from lab import engine, plan, runs, speeds
from lab.gpu_monitor import GpuMonitor

WARM_UP_SECONDS = 2.0


def image_digest(folder):
    digest = hashlib.sha256()
    for path in sorted(glob.glob(os.path.join(folder, "*.bin"))):
        with open(path, "rb") as file:
            for block in iter(lambda: file.read(1 << 24), b""):
                digest.update(block)
    return digest.hexdigest()[:16]


def run_point(each, scratch_folder):
    shutil.rmtree(scratch_folder, ignore_errors=True)
    command = engine.engine_command(plan.star_config(each), scratch_folder, each["processor"], each["precision"], None)
    started_at = time.perf_counter()
    with GpuMonitor() as monitor:
        output = engine.engine_output(command)
    process_time = time.perf_counter() - started_at
    result = dict(each, IMAGE_DIGEST=image_digest(scratch_folder), PROCESS_TIME=process_time,
                  SUMMARY=engine.lines_of_kind(output, "SUMMARY")[0], KERNEL=engine.lines_of_kind(output, "KERNEL"),
                  SMS=engine.lines_of_kind(output, "SMS"), SECTIONS=engine.lines_of_kind(output, "SECTION"),
                  GPU_MONITOR=monitor.summary())
    shutil.rmtree(scratch_folder, ignore_errors=True)
    if each["variant"] is None:
        speeds.remember_engine_run(result["SUMMARY"])
    return result


def point_path(folder, name):
    return os.path.join(folder, "points", name + ".json")


def done(folder, name):
    return os.path.exists(point_path(folder, name))


def read(folder, name):
    return runs.read_json(point_path(folder, name))


def warm_up(precision, scratch_folder):
    photons = int(WARM_UP_SECONDS * speeds.speed_of("gpu", precision)[0])
    run_point(plan.point("gpu", precision, photons), scratch_folder)


def log(folder, text):
    print(text, flush=True)
    with open(os.path.join(folder, "progress.log"), "a") as file:
        file.write(f"{runs.now()} {text}\n")


def run_missing(the_plan, folder):
    os.makedirs(os.path.join(folder, "points"), exist_ok=True)
    scratch_folder = os.path.join(folder, "scratch")
    missing = [entry for entry in the_plan.points() if not done(folder, entry["key"])]
    left = sum(entry["prediction"]["seconds"] for entry in missing)
    log(folder, f"{len(the_plan.points()) - len(missing)} of {len(the_plan.points())} runs already done; "
                f"{len(missing)} to go, ~{plan.duration(left)}")
    warmed = False
    for number, entry in enumerate(missing, 1):
        each = entry["point"]
        if each["processor"] == "gpu" and not warmed:
            warm_up(each["precision"], scratch_folder)
            warmed = True
        result = run_point(each, scratch_folder)
        result.update(parts=entry["parts"], validity=entry["validity"], predicted_seconds=entry["prediction"]["seconds"])
        runs.save_json(result, point_path(folder, entry["key"]))
        left -= entry["prediction"]["seconds"]
        log(folder, f"  {number}/{len(missing)} {entry['key']}: {result['PROCESS_TIME']:.1f} s "
                    f"(predicted {entry['prediction']['seconds']:.1f} s); ~{plan.duration(left)} left")
    shutil.rmtree(scratch_folder, ignore_errors=True)
