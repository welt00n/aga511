import importlib.util
import os
import shutil
import subprocess
import sys

from lab import device, engine, gpu_monitor, plan, runs, speeds

OK, WARN, FAIL = "ok", "warn", "FAIL"
REPORT_PACKAGES = ("numpy", "matplotlib", "pandas", "nbformat", "nbclient", "nbconvert")
FREE_DISK_WANTED = 2e9


def check(status, name, detail):
    return {"status": status, "name": name, "detail": detail}


def tool(name):
    beside_python = os.path.join(os.path.dirname(sys.executable), name)
    return shutil.which(name) or (beside_python if os.path.exists(beside_python) else None)


# ============================================================ the checks

def compiler():
    if not os.path.exists(engine.NVCC):
        return check(FAIL, "nvcc", f"not found at {engine.NVCC}")
    return check(OK, "nvcc", engine.nvcc_version().strip().splitlines()[-1])


def gpu():
    if not gpu_monitor.gpu_is_visible():
        return check(FAIL, "GPU", "nvidia-smi not found: no GPU visible")
    name = gpu_monitor.query_now(["name", "driver_version", "memory.free"])
    if not name:
        return check(FAIL, "GPU", "nvidia-smi found but it can't read the GPU")
    return check(OK, "GPU", f"{name[0]}, driver {name[1]}, {float(name[2]) / 1024:.1f} GiB free")


def other_gpu_work():
    processes = gpu_monitor.compute_processes()
    if processes:
        return check(WARN, "other GPU work", f"{len(processes)} other process(es) on the GPU: they would skew the "
                                             f"times ({'; '.join(processes)})")
    return check(OK, "other GPU work", "none")


def gpu_power():
    values = gpu_monitor.query_now(["clocks.sm", "power.draw", "enforced.power.limit", "temperature.gpu"])
    if not values:
        return check(WARN, "GPU power", "not readable")
    return check(OK, "GPU power", f"now {values[0]} MHz, {values[1]} W of a {values[2]} W limit, {values[3]} °C "
                                  "(read only: clocks and limits are not changed)")


def disk():
    os.makedirs(runs.RUNS_FOLDER, exist_ok=True)
    free = shutil.disk_usage(runs.RUNS_FOLDER).free
    status = OK if free >= FREE_DISK_WANTED else WARN
    return check(status, "disk", f"{free / 1e9:.0f} GB free under runs/")


def packages(names):
    missing = [name for name in names if importlib.util.find_spec(name) is None]
    if missing:
        return check(FAIL, "Python packages", "missing: " + ", ".join(missing))
    return check(OK, "Python packages", ", ".join(names))


def pdf_engine():
    if tool("tectonic") and tool("pandoc"):
        return check(OK, "PDF", "tectonic and pandoc")
    return check(WARN, "PDF", "tectonic or pandoc missing: the report will be HTML instead of PDF")


def video():
    if tool("ffmpeg"):
        return check(OK, "video", "ffmpeg: mp4")
    return check(FAIL, "video", "no ffmpeg: install it from conda-forge (the videos are mp4)")


def committed_engine():
    result = subprocess.run(["git", "status", "--porcelain", "--", engine.ENGINE_FOLDER], capture_output=True,
                            text=True, cwd=engine.EP4_FOLDER)
    if result.returncode != 0:
        return check(WARN, "engine code", "not in a git repository: the code hash is the only record")
    changed = [line for line in result.stdout.splitlines() if line.strip()]
    if changed:
        return check(WARN, "engine code", f"{len(changed)} uncommitted change(s): code {engine.code_hash()} "
                                          "matches no commit")
    return check(OK, "engine code", f"committed (code {engine.code_hash()})")


def measured_speeds(binaries):
    guessed = [f"{processor}{precision}" for processor, precision in binaries
               if speeds.measured_speed(speeds.binary_key(processor, precision)) is None]
    if guessed:
        return check(WARN, "speeds", f"not measured here yet for {', '.join(guessed)}: predictions use the laptop's "
                                     "figures (run `test` first, or accept rougher predictions)")
    return check(OK, "speeds", "measured on this machine")


# ============================================================ what each product needs

def checks_for(needs_gpu=True, needs_report=False, needs_video=False):
    binaries = [("gpu", precision) for precision in engine.PRECISIONS]
    found = [compiler(), disk(), committed_engine()]
    if needs_gpu:
        found += [gpu(), other_gpu_work(), gpu_power()]
    found.append(packages(REPORT_PACKAGES if needs_report else ("numpy", "matplotlib")))
    if needs_report:
        found.append(pdf_engine())
    if needs_video:
        found.append(video())
    found.append(measured_speeds(binaries))
    return found


# ============================================================ the GPU and the launch the engine will use

def gpu_line(limits, launch):
    warps = limits["MAX_THREADS_PER_SM"] // limits["WARP_SIZE"]
    return (f"  {limits['NAME']}, cc {launch['COMPUTE_CAPABILITY']}, {launch['COMPUTE_UNITS_TOTAL']} SMs; per SM: "
            f"{limits['MAX_THREADS_PER_SM']} threads ({warps} warps of {limits['WARP_SIZE']}), "
            f"{limits['REGISTERS_PER_SM']} registers, at most {limits['MAX_BLOCKS_PER_SM']} blocks")


def spaced(number):
    return f"{number:,}".replace(",", " ")


def launch_line(precision, limits, launch):
    threads_per_sm = launch["THREADS_PER_BLOCK"] * launch["BLOCKS_PER_UNIT"]
    warps = threads_per_sm // limits["WARP_SIZE"]
    allowed = min(device.warps_allowed_by_registers(launch["REGISTERS_PER_THREAD"], limits), device.max_warps(limits))
    whole = " (blocks fit only whole)" if warps < allowed else ""
    return (f"  FP{precision}: {launch['REGISTERS_PER_THREAD']} registers per thread (they allow {allowed} warps per SM) "
            f"→ {launch['THREADS_PER_BLOCK']} threads per block × {launch['BLOCKS_PER_UNIT']} block(s) per SM = "
            f"{warps} warps per SM{whole}, occupancy {launch['OCCUPANCY']:.2f}; "
            f"{spaced(threads_per_sm * launch['COMPUTE_UNITS_TOTAL'])} threads in all")


def launch_lines(limits, launches):
    return (["\nThe GPU and the launch (the occupancy API sizes the blocks from the registers; one grid fills "
             "every SM once):", gpu_line(limits, launches[64])]
            + [launch_line(precision, limits, launch) for precision, launch in launches.items()])


def show_launch():
    if not gpu_monitor.gpu_is_visible():
        return
    limits = device.limits()
    launches = {precision: plan.described("gpu", precision, None, None, "{}") for precision in engine.PRECISIONS}
    if limits:
        print("\n".join(launch_lines(limits, launches)))


def show(found):
    print("\nThe environment:")
    for each in found:
        print(f"  [{each['status']:>4s}] {each['name']}: {each['detail']}")
    return not any(each["status"] == FAIL for each in found)
