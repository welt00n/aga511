import functools
import math
import os
import re
import subprocess
import tempfile

from lab import engine

REGISTER_STEP = 8
LIMITS_SOURCE = r"""
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) return 1;
    printf("NAME=%s\n", p.name);
    printf("COMPUTE_CAPABILITY=%d%d\n", p.major, p.minor);
    printf("SMS=%d\n", p.multiProcessorCount);
    printf("REGISTERS_PER_SM=%d\n", p.regsPerMultiprocessor);
    printf("MAX_THREADS_PER_SM=%d\n", p.maxThreadsPerMultiProcessor);
    printf("MAX_BLOCKS_PER_SM=%d\n", p.maxBlocksPerMultiProcessor);
    printf("WARP_SIZE=%d\n", p.warpSize);
    printf("SHARED_MEMORY_PER_SM=%zu\n", p.sharedMemPerMultiprocessor);
    return 0;
}
"""
KNOWN_TARGETS = {
    "80": {"NAME": "A30 (NVIDIA's Ampere tuning guide)", "COMPUTE_CAPABILITY": 80, "SMS": 56,
           "REGISTERS_PER_SM": 65536, "MAX_THREADS_PER_SM": 2048, "MAX_BLOCKS_PER_SM": 32, "WARP_SIZE": 32},
}
KERNEL_ENTRY = "Compiling entry function '_Z13trace_photons"


# ============================================================ this GPU's limits

def parse_limits(text):
    pairs = (line.split("=", 1) for line in text.splitlines() if "=" in line)
    return {key: value if key == "NAME" else int(value) for key, value in pairs}


@functools.cache
def limits():
    binary = os.path.join(engine.BUILD_FOLDER, "device_limits")
    if not os.path.exists(binary):
        os.makedirs(engine.BUILD_FOLDER, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", suffix=".cu", delete=False) as source:
            source.write(LIMITS_SOURCE)
        subprocess.run([engine.NVCC, source.name, "-o", binary], check=True, capture_output=True)
        os.remove(source.name)
    result = subprocess.run([binary], capture_output=True, text=True)
    return parse_limits(result.stdout) if result.returncode == 0 else None


# ============================================================ registers → warps → occupancy

def registers_per_warp(registers, device):
    return math.ceil(registers / REGISTER_STEP) * REGISTER_STEP * device["WARP_SIZE"]


def max_warps(device):
    return device["MAX_THREADS_PER_SM"] // device["WARP_SIZE"]


def warps_allowed_by_registers(registers, device):
    return device["REGISTERS_PER_SM"] // registers_per_warp(registers, device)


def occupancy_allowed_by_registers(registers, device):
    return min(warps_allowed_by_registers(registers, device), max_warps(device)) / max_warps(device)


def occupancy_row(registers, spills, device):
    warps = min(warps_allowed_by_registers(registers, device), max_warps(device))
    return {"registers": registers, "spill_bytes": spills,
            "registers_per_warp": registers_per_warp(registers, device),
            "warps_allowed_by_registers": warps_allowed_by_registers(registers, device),
            "max_warps": max_warps(device), "threads_per_sm": warps * device["WARP_SIZE"],
            "occupancy_at_most": occupancy_allowed_by_registers(registers, device)}


# ============================================================ another GPU, compile only

def kernel_resources(report):
    part = report[report.find(KERNEL_ENTRY):]
    registers = re.search(r"Used (\d+) registers", part)
    spills = re.search(r"(\d+) bytes spill stores", part)
    return int(registers.group(1)), int(spills.group(1)) if spills else 0


def spill_bytes(precision, variant=None):
    report = engine.read_text(engine.binary_path("gpu", precision, variant) + ".report") or ""
    return kernel_resources(report)[1] if KERNEL_ENTRY in report else None


def compiled_resources(precision, architecture):
    with tempfile.TemporaryDirectory() as folder:
        command = [engine.NVCC, *engine.FLAGS, *engine.architecture_flags(architecture),
                   *(["-DSTAR_FP32"] if precision == 32 else []), "-cubin", "-Xptxas", "-v", "gpu/gpu.cu",
                   "-o", os.path.join(folder, "gpu.cubin")]
        result = subprocess.run(command, cwd=engine.ENGINE_FOLDER, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"compiling for sm_{architecture} failed:\n{result.stderr}")
    return kernel_resources(result.stderr)


def target_occupancy(architecture):
    device = KNOWN_TARGETS[architecture]
    rows = {}
    for precision in engine.PRECISIONS:
        registers, spills = compiled_resources(precision, architecture)
        rows[f"fp{precision}"] = occupancy_row(registers, spills, device)
    return {"target": f"sm_{architecture}", "device": device, "precisions": rows}
