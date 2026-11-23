import collections

import common

FP64_OPCODE = ("DFMA", "DMUL", "DADD", "DSETP", "DMNMX", "MUFU.RSQ64H", "MUFU.RCP64H")


def nop_runs(opcodes):
    runs, start = [], None
    for index, name in enumerate(opcodes + ["END"]):
        if name == "NOP" and start is None:
            start = index
        elif name != "NOP" and start is not None:
            runs.append((index - start, opcodes[start - 1] if start else "START", name))
            start = None
    return runs


def is_fp64(name):
    return name.startswith(FP64_OPCODE)


def describe(precision, binary):
    opcodes = [common.opcode(instruction) for instruction in common.kernel_sass(binary)]
    runs = nop_runs(opcodes)
    nops = sum(length for length, _, _ in runs)
    print(f"== FP{precision}: {len(opcodes)} instructions, {nops} NOPs in {len(runs)} runs")
    print("  run length: " + ", ".join(f"{length} x{count}" for length, count in
                                       sorted(collections.Counter(length for length, _, _ in runs).items())))
    print("  instruction before a run: " + ", ".join(f"{name} {count}" for name, count in
                                                     collections.Counter(before for _, before, _ in runs).most_common(6)))
    print("  instruction after a run:  " + ", ".join(f"{name} {count}" for name, count in
                                                     collections.Counter(after for _, _, after in runs).most_common(6)))
    between = sum(length for length, before, after in runs if is_fp64(before) and is_fp64(after))
    print(f"  NOPs between two FP64 instructions: {between} of {nops}")
    tail = runs[-1] if runs and runs[-1][2] == "END" else None
    print(f"  NOPs after the last instruction (end padding): {tail[0] if tail else 0}")


with common.recorded(__file__):
    folder = common.scratch_folder("nop_runs")
    print("Q2: where are the kernel's NOPs, and what surrounds them?\n")
    for precision in (64, 32):
        describe(precision, common.kernel_cubin(precision, folder))
    print("\nFound on 2026-11-22: FP64 has 1063 NOPs in 304 runs, 233 of them 4 long, mostly right before a DFMA,")
    print("DMUL or DADD that needs the previous FP64 result: the compiler makes the warp wait for the slow FP64")
    print("unit. They are executed, not padding (only 9 sit at the end). FP32 has 87, mostly single, before branches.")
