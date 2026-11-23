import collections
import os
import re

import common
from lab import engine

COMPUTES = (75, 80, 86, 89, 90, 100, 103, 110, 120)
COMPARED = (90, 120)
BRANCH = re.compile(r"^\s*(@%?!?\w+\s+)?bra")
SELECT = re.compile(r"^\s*selp")
PTX_OPCODE = re.compile(r"^\s*(?:@%?!?\w+\s+)?([a-z][a-z0-9_]*(?:\.[a-z0-9_]+)*)\s")


def ptx_opcodes(path):
    counts = collections.Counter()
    for line in open(path):
        if line.strip().startswith(("//", ".", "{", "}", "$")):
            continue
        found = PTX_OPCODE.match(line)
        if found:
            counts[found.group(1)] += 1
    return counts


def count_lines(path, pattern):
    return sum(1 for line in open(path) if pattern.match(line))


def ptx_or_none(precision, folder, compute):
    try:
        return common.kernel_ptx(precision, folder, compute)
    except RuntimeError:
        return None


def registers_or_dash(ptx, architecture):
    try:
        return str(common.ptxas_registers(ptx, architecture)[0])
    except RuntimeError:
        return "-"


def registers_by_front_end(precision, folder):
    print(f"== FP{precision}: PTX made for compute_XX, then ptxas for sm_120 and for sm_XX")
    print(f"  {'PTX':12s} {'on sm_120':>10s} {'on sm_XX':>9s} {'branches':>9s} {'selects':>8s}  PTX version")
    for compute in COMPUTES:
        ptx = ptx_or_none(precision, folder, compute)
        if ptx is None:
            print(f"  compute_{compute}: not supported by this nvcc")
            continue
        version = re.search(r"\.version ([\d.]+)", open(ptx).read()).group(1)
        print(f"  compute_{compute:<4d} {registers_or_dash(ptx, 120):>10s} {registers_or_dash(ptx, compute):>9s} "
              f"{count_lines(ptx, BRANCH):9d} "
              f"{count_lines(ptx, SELECT):8d}  {version}")


def opcode_difference(precision, folder):
    old, new = (ptx_opcodes(common.kernel_ptx(precision, folder, compute)) for compute in COMPARED)
    print(f"  opcodes that differ most, compute_{COMPARED[0]} -> compute_{COMPARED[1]}:")
    for name in sorted(set(old) | set(new), key=lambda name: -abs(new[name] - old[name]))[:12]:
        if new[name] != old[name]:
            print(f"    {name:28s} {old[name]:5d} -> {new[name]:5d} ({new[name] - old[name]:+d})")


def nvvm_latest_is_accepted(folder):
    try:
        common.compile_kernel(64, os.path.join(folder, "latest.cubin"), ["-arch=sm_80", "-cubin", "-nvvm-latest"])
        return True
    except RuntimeError:
        return False


with common.recorded(__file__):
    print("Why does sm_80 need more registers than sm_120? Separate the front end (C++ -> PTX, NVVM) from the")
    print("back end (PTX -> SASS, ptxas): compile the PTX for each compute_XX and assemble it for each SM.\n")
    print(common.run([engine.NVCC, "--version"]).strip().splitlines()[-1] + "\n")
    folder = common.scratch_folder("front_end")
    for precision in (64, 32):
        registers_by_front_end(precision, folder)
        opcode_difference(precision, folder)
        print()
    print(f"nvcc accepts -nvvm-latest (the newer front end for older targets): {nvvm_latest_is_accepted(folder)}")
    print("\nFound on 2026-11-22: the jump is at compute_100. PTX for compute_75...90 needs 96-98 registers (FP64),")
    print("compute_100...120 80-84, whatever SM ptxas assembles it for. CUDA 13 uses a newer NVVM (LLVM 18) for")
    print("Blackwell and later; its PTX has fewer branches (FP64 107 -> 90, FP32 112 -> 91). ptxas adds only ~2-5.")
