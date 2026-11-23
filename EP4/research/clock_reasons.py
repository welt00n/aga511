import subprocess
import threading

import common
from lab import engine
from lab.gpu_monitor import NVIDIA_SMI

FIELDS = {"MHz": "clocks.sm", "max MHz": "clocks.max.sm", "W": "power.draw", "limit W": "enforced.power.limit",
          "°C": "temperature.gpu", "power cap": "clocks_event_reasons.sw_power_cap",
          "sw thermal": "clocks_event_reasons.sw_thermal_slowdown", "hw slow": "clocks_event_reasons.hw_slowdown",
          "hw thermal": "clocks_event_reasons.hw_thermal_slowdown",
          "power brake": "clocks_event_reasons.hw_power_brake_slowdown",
          "board": "clocks_event_reasons.board_limit", "reliability": "clocks_event_reasons.reliability"}
CASES = [("FP32, 36 SMs", 2e11, 32, None), ("FP64, 36 SMs", 1.2e10, 64, None), ("FP32, 1 SM", 6e9, 32, 1)]
BUSY_WATTS = 25
ROWS_SHOWN = 6


def sample(samples, stop):
    command = [NVIDIA_SMI, "--query-gpu=" + ",".join(FIELDS.values()), "--format=csv,noheader,nounits", "-lms=250"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        samples.append([value.strip() for value in line.split(",")])
        if stop.is_set():
            break
    process.terminate()


def is_busy(row):
    try:
        return len(row) == len(FIELDS) and float(row[2]) > BUSY_WATTS
    except ValueError:
        return False


def measure(label, photons, precision, sms):
    samples, stop = [], threading.Event()
    sampler = threading.Thread(target=sample, args=(samples, stop), daemon=True)
    sampler.start()
    summary = common.engine_summary(engine.default_star_config(photons, sms=sms), "gpu", precision)
    stop.set()
    sampler.join(timeout=2)
    busy = [row for row in samples if is_busy(row)]
    print(f"== {label}: {photons / summary['PHOTONS_TIME']:.2e} photons/s, {len(busy)} busy samples of {len(samples)}")
    print("   " + " ".join(f"{name:>11s}" for name in FIELDS))
    for row in busy[:: max(1, len(busy) // ROWS_SHOWN)]:
        print("   " + " ".join(f"{value:>11s}" for value in row))


with common.recorded(__file__):
    print("Why does the GPU's clock fall? The driver's own reasons (nvidia-smi clocks_event_reasons) during runs.\n")
    for case in CASES:
        measure(*case)
    print("\nFound on 2026-11-22: FP64, and FP32 on 1 SM, run at ~2880-2895 MHz, held by 'reliability' (the voltage")
    print("ceiling). FP32 on 36 SMs hits the software power cap: the enforced limit moves between 85 and 115 W")
    print("(Dynamic Boost) and the clock follows it down. Register spills are not the cause.")
