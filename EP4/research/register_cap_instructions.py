import collections
import os
import re

import common
from lab import engine

CAPS = {64: [None, 72, 64, 56, 48, 40, 32], 32: [None, 64, 56, 48, 40, 32]}
OPCODE = re.compile(r"^(?:@!?U?P[0-9T]\s+)?([A-Z][A-Z0-9_]*)")


def counts_of(binary):
    counts = collections.Counter()
    for instruction in common.kernel_sass(binary):
        found = OPCODE.match(instruction)
        if found:
            counts[found.group(1)] += 1
    return counts


def built(precision, cap):
    variant = f"registers_{cap}" if cap else None
    if variant:
        engine.GPU_VARIANTS[variant] = [f"-maxrregcount={cap}"]
    return engine.build_engine("gpu", precision, variant=variant)


with common.recorded(__file__):
    print("What does a register cap do to the kernel's code? The machine instructions of trace_photons (static: in the")
    print("code, not executed) for each -maxrregcount cap: all of them without the NOPs, and the spill stores and loads")
    print("(STL / LDL: local memory, which is private to the thread but lives in device memory, cached by L1 and L2).\n")
    for precision, caps in CAPS.items():
        print(f"== FP{precision}")
        base = None
        for cap in caps:
            binary = built(precision, cap)
            counts = counts_of(binary)
            real = sum(count for opcode, count in counts.items() if opcode != "NOP")
            base = base or real
            registers, spills = common.kernel_resources(engine.read_text(binary + ".report"))
            print(f"  cap {str(cap or 'none'):>4}: {registers:3d} registers, {spills:3d} B spilled, {real:5d} instructions "
                  f"({real / base:.3f}×), STL {counts['STL']:3d}, LDL {counts['LDL']:3d}")
    print("\nFound on 2026-11-22: fewer registers mean more instructions even before anything spills (FP32 at 55")
    print("registers, 0 B spilled: +4.9%): the compiler recomputes values instead of keeping them and overlaps less.")
    print("Then the spills add stores and loads (FP64 at 40 registers: 68 STL, 48 LDL, +15% instructions).")
