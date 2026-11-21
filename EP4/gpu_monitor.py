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
    "TEMPERATURE_C": "temperature.gpu",
    "UTILIZATION_PERCENT": "utilization.gpu",
    "MEMORY_USED_MIB": "memory.used",
}


def number_or_none(text):
    try:
        return float(text)
    except ValueError:
        return None


class GpuMonitor:
    def __init__(self, interval_ms=200):
        self.interval_ms = interval_ms
        self.samples = []

    def __enter__(self):
        command = [NVIDIA_SMI, "--query-gpu=" + ",".join(QUERY.values()), "--format=csv,noheader,nounits",
                   f"-lms={self.interval_ms}"]
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self.reader = threading.Thread(target=self.read_samples, daemon=True)
        self.reader.start()
        return self

    def read_samples(self):
        for line in self.process.stdout:
            values = [number_or_none(value.strip()) for value in line.split(",")]
            if len(values) == len(QUERY):
                self.samples.append(dict(zip(QUERY, values)))

    def __exit__(self, *exception):
        self.process.terminate()
        self.process.wait()
        self.reader.join(timeout=1)

    def summary(self):
        summary = {"GPU_SAMPLES": len(self.samples)}
        for key in QUERY:
            values = [sample[key] for sample in self.samples if sample[key] is not None]
            if values:
                summary[key + "_MIN"] = min(values)
                summary[key + "_MEDIAN"] = statistics.median(values)
                summary[key + "_MAX"] = max(values)
        return summary


def gpu_is_visible():
    return os.path.exists(NVIDIA_SMI)
