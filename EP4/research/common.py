import contextlib
import datetime
import functools
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile

RESEARCH_FOLDER = os.path.dirname(os.path.abspath(__file__))
EP4_FOLDER = os.path.dirname(RESEARCH_FOLDER)
RESULTS_FOLDER = os.path.join(RESEARCH_FOLDER, "results")
sys.path.insert(0, EP4_FOLDER)

from lab import engine  # noqa: E402

CUDA_BIN = os.path.dirname(engine.NVCC)
KERNEL = "trace_photons"
SM_LIMITS_SOURCE = r"""
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("NAME=%s\n", p.name);
    printf("COMPUTE_CAPABILITY=%d%d\n", p.major, p.minor);
    printf("SMS=%d\n", p.multiProcessorCount);
    printf("REGISTERS_PER_SM=%d\n", p.regsPerMultiprocessor);
    printf("REGISTERS_PER_BLOCK=%d\n", p.regsPerBlock);
    printf("MAX_THREADS_PER_SM=%d\n", p.maxThreadsPerMultiProcessor);
    printf("MAX_THREADS_PER_BLOCK=%d\n", p.maxThreadsPerBlock);
    printf("MAX_BLOCKS_PER_SM=%d\n", p.maxBlocksPerMultiProcessor);
    printf("WARP_SIZE=%d\n", p.warpSize);
    printf("SHARED_MEMORY_PER_SM=%zu\n", p.sharedMemPerMultiprocessor);
    return 0;
}
"""


def tool(name):
    return os.path.join(CUDA_BIN, name)


def run(command, cwd=None, check=True):
    result = subprocess.run([str(part) for part in command], cwd=cwd, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"{' '.join(map(str, command))} failed:\n{result.stdout}{result.stderr}")
    return result.stdout + result.stderr


def scratch_folder(name):
    return tempfile.mkdtemp(prefix=f"ep4_research_{name}_")


def precision_flags(precision):
    return ["-DSTAR_FP32"] if precision == 32 else []


def compile_kernel(precision, output, flags):
    command = [engine.NVCC, *engine.FLAGS, *precision_flags(precision), *flags, "gpu/gpu.cu", "-o", output]
    return run(command, cwd=engine.ENGINE_FOLDER)


def kernel_cubin(precision, folder, architecture="native", flags=()):
    output = os.path.join(folder, f"gpu{precision}_{architecture}.cubin")
    arch = ["-arch=native"] if architecture == "native" else [f"-arch=sm_{architecture}"]
    compile_kernel(precision, output, [*arch, "-lineinfo", "-cubin", *flags])
    return output


def kernel_ptx(precision, folder, compute):
    output = os.path.join(folder, f"gpu{precision}_compute_{compute}.ptx")
    compile_kernel(precision, output, [f"-arch=compute_{compute}", "-lineinfo", "-ptx"])
    return output


def ptxas_registers(ptx, architecture):
    report = run([tool("ptxas"), "-O3", "-v", f"-arch=sm_{architecture}", ptx, "-o", os.devnull])
    return kernel_resources(report)


def kernel_resources(report):
    part = report[report.find(f"Compiling entry function '_Z{len(KERNEL)}{KERNEL}"):]
    registers = re.search(r"Used (\d+) registers", part)
    spills = re.search(r"(\d+) bytes spill stores", part)
    return int(registers.group(1)), int(spills.group(1)) if spills else 0


SASS_LINE = re.compile(r"^\s*/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;")


def kernel_sass(binary):
    instructions, inside = [], False
    for line in run([tool("cuobjdump"), "-sass", binary]).splitlines():
        if "Function :" in line:
            inside = KERNEL in line
            continue
        found = SASS_LINE.match(line)
        if inside and found:
            instructions.append(found.group(2))
    return instructions


def opcode(instruction):
    parts = instruction.split()
    return parts[1] if parts[0].startswith("@") else parts[0]


def engine_output(config, processor, precision, prefix=()):
    folder = scratch_folder("engine")
    try:
        return engine.engine_output([*prefix, *engine.engine_command(config, folder, processor, precision, None)])
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def engine_summary(config, processor, precision, prefix=()):
    return engine.lines_of_kind(engine_output(config, processor, precision, prefix), "SUMMARY")[0]


@functools.cache
def sm_limits():
    folder = scratch_folder("sm_limits")
    source, binary = os.path.join(folder, "sm_limits.cu"), os.path.join(folder, "sm_limits")
    with open(source, "w") as file:
        file.write(SM_LIMITS_SOURCE)
    run([engine.NVCC, source, "-o", binary])
    output = run([binary])
    shutil.rmtree(folder)
    pairs = (line.split("=", 1) for line in output.splitlines() if "=" in line)
    return {key: value if key == "NAME" else int(value) for key, value in pairs}


REGISTER_STEP = 8
A30_LIMITS = {"NAME": "A30 (Ampere tuning guide)", "COMPUTE_CAPABILITY": 80, "SMS": 56,
              "REGISTERS_PER_SM": 65536, "MAX_THREADS_PER_SM": 2048, "MAX_BLOCKS_PER_SM": 32, "WARP_SIZE": 32}


def registers_per_warp(registers, limits):
    return math.ceil(registers / REGISTER_STEP) * REGISTER_STEP * limits["WARP_SIZE"]


def warps_allowed_by_registers(registers, limits):
    return limits["REGISTERS_PER_SM"] // registers_per_warp(registers, limits)


def max_warps(limits):
    return limits["MAX_THREADS_PER_SM"] // limits["WARP_SIZE"]


def occupancy_allowed_by_registers(registers, limits):
    return min(warps_allowed_by_registers(registers, limits), max_warps(limits)) / max_warps(limits)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def header(script):
    nvcc = run([engine.NVCC, "--version"]).strip().splitlines()[-1]
    return (f"# {os.path.basename(script)}\n"
            f"# {datetime.datetime.now():%Y-%m-%d %H:%M} on {socket.gethostname()}, "
            f"engine code {engine.code_hash()}, {nvcc}\n")


@contextlib.contextmanager
def recorded(script):
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    path = os.path.join(RESULTS_FOLDER, os.path.splitext(os.path.basename(script))[0] + ".txt")
    with open(path, "w") as file:
        terminal, sys.stdout = sys.stdout, Tee(sys.stdout, file)
        try:
            print(header(script))
            yield
        finally:
            sys.stdout = terminal
    print(f"\n(saved in {os.path.relpath(path, EP4_FOLDER)})")
