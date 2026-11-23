import collections
import os
import re

import common

FP64_SASS = re.compile(r"^(D[A-Z]+|[FI]2[FI]\.[A-Z0-9.]*64[A-Z0-9.]*|MUFU\.[A-Z]+\.64)")
PTX_FILE = re.compile(r'^\s*\.file\s+(\d+)\s+"([^"]+)"')
PTX_LOC = re.compile(r"^\s*\.loc\s+(\d+)\s+(\d+)\s+\d+(.*)")
INLINED_AT = re.compile(r"inlined_at\s+(\d+)\s+(\d+)")
FP64_PTX = re.compile(r"\.f64\b")


def fp64_sass_instructions(binary):
    return [instruction for instruction in common.kernel_sass(binary) if FP64_SASS.match(common.opcode(instruction))]


def ptx_files(lines):
    return {found.group(1): os.path.basename(found.group(2)) for found in map(PTX_FILE.match, lines) if found}


def fp64_ptx_sites(ptx):
    lines = open(ptx).read().splitlines()
    files, where, sites = ptx_files(lines), None, collections.Counter()
    for line in lines:
        found = PTX_LOC.match(line)
        if found:
            chain = [(found.group(1), found.group(2)), *INLINED_AT.findall(found.group(3))]
            where = " <- ".join(f"{files.get(index, index)}:{number}" for index, number in chain)
            continue
        if FP64_PTX.search(line) and not line.strip().startswith(("//", ".")):
            sites[where] += 1
    return sites


def double_precision_warnings(folder):
    report = common.compile_kernel(32, os.path.join(folder, "warn.cubin"),
                                   ["-arch=native", "-cubin", "-Xptxas", "--warn-on-double-precision-use"])
    warnings = [line for line in report.splitlines() if "double" in line.lower()]
    lines = sorted({int(found.group(1)) for found in map(re.compile(r"line (\d+);").search, warnings) if found})
    return f"{len(warnings)} warnings, at PTX lines {lines}" if warnings else "no warning"


with common.recorded(__file__):
    folder = common.scratch_folder("fp64_in_fp32")
    print("Q1: which instructions in the FP32 kernel are FP64, and where do they come from?\n")
    print(f"ptxas --warn-on-double-precision-use: {double_precision_warnings(folder)}")
    instructions = fp64_sass_instructions(common.kernel_cubin(32, folder))
    print(f"\nFP64 instructions in the FP32 kernel's SASS: {len(instructions)}")
    for instruction in instructions:
        print("  " + instruction)
    ptx = common.kernel_ptx(32, folder, common.sm_limits()["COMPUTE_CAPABILITY"])
    print("\nPTX .f64 operations by source line (innermost first, then where it was inlined):")
    for site, count in fp64_ptx_sites(ptx).most_common():
        print(f"  {count:3d}  {site}")
    print("\nFound on 2026-11-22: 12 instructions, 3 per call (I2F.F64 -> DMUL by a constant -> F2F.F32), in the 4")
    print("sinf/cosf calls (real.cuh:26/28, used at photon.cuh:36-37 and 56-57). They are CUDA's Payne-Hanek")
    print("argument reduction, taken only for |x| > ~1e5 radians: never for the star's angles.")
