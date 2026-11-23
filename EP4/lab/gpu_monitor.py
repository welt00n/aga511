import functools
import os
import shutil
import statistics
import subprocess
import threading

NVIDIA_SMI = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
QUERY = {
    "SM_CLOCK_MHZ": "clocks.sm",
    "MEMORY_CLOCK_MHZ": "clocks.mem",
    "POWER_W": "power.draw",
    "POWER_LIMIT_W": "enforced.power.limit",
    "TEMPERATURE_C": "temperature.gpu",
    "UTILIZATION_PERCENT": "utilization.gpu",
    "MEMORY_USED_MIB": "memory.used",
    "POWER_CAP": "clocks_event_reasons.sw_power_cap",
    "VOLTAGE_CEILING": "clocks_event_reasons.reliability",
    "THERMAL_SLOWDOWN": "clocks_event_reasons.sw_thermal_slowdown",
    "HARDWARE_SLOWDOWN": "clocks_event_reasons.hw_slowdown",
}
FLAGS = ("POWER_CAP", "VOLTAGE_CEILING", "THERMAL_SLOWDOWN", "HARDWARE_SLOWDOWN")
FLAG_VALUES = {"Active": 1.0, "Not Active": 0.0}


def number_or_none(text):
    if text in FLAG_VALUES:
        return FLAG_VALUES[text]
    try:
        return float(text)
    except ValueError:
        return None


def field_works(field):
    result = subprocess.run([NVIDIA_SMI, f"--query-gpu={field}", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True)
    return result.returncode == 0


@functools.cache
def supported_query():
    if not gpu_is_visible():
        return {}
    if field_works(",".join(QUERY.values())):
        return dict(QUERY)
    return {key: field for key, field in QUERY.items() if field_works(field)}


class GpuMonitor:
    def __init__(self, interval_ms=200):
        self.interval_ms = interval_ms
        self.samples = []
        self.query = supported_query()

    def __enter__(self):
        if not self.query:
            self.process = None
            return self
        command = [NVIDIA_SMI, "--query-gpu=" + ",".join(self.query.values()), "--format=csv,noheader,nounits",
                   f"-lms={self.interval_ms}"]
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self.reader = threading.Thread(target=self.read_samples, daemon=True)
        self.reader.start()
        return self

    def read_samples(self):
        for line in self.process.stdout:
            values = [number_or_none(value.strip()) for value in line.split(",")]
            if len(values) == len(self.query):
                self.samples.append(dict(zip(self.query, values)))

    def __exit__(self, *exception):
        if self.process is None:
            return
        self.process.terminate()
        self.process.wait()
        self.reader.join(timeout=1)

    def summary(self):
        summary = {"GPU_SAMPLES": len(self.samples)}
        for key in self.query:
            values = [sample[key] for sample in self.samples if sample[key] is not None]
            if not values:
                continue
            if key in FLAGS:
                summary[key + "_SHARE"] = sum(values) / len(values)
            else:
                summary[key + "_MIN"] = min(values)
                summary[key + "_MEDIAN"] = statistics.median(values)
                summary[key + "_MAX"] = max(values)
        return summary


def gpu_is_visible():
    return os.path.exists(NVIDIA_SMI)


def query_now(fields):
    result = subprocess.run([NVIDIA_SMI, "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"],
                            capture_output=True, text=True)
    return [value.strip() for value in result.stdout.strip().split(",")] if result.returncode == 0 else None


def compute_processes():
    result = subprocess.run([NVIDIA_SMI, "--query-compute-apps=pid,process_name,used_memory",
                             "--format=csv,noheader"], capture_output=True, text=True)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()] if result.returncode == 0 else []
