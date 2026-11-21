import argparse
import collections
import os
import re
import subprocess
from datetime import datetime

import engine

CUOBJDUMP = "/usr/local/cuda/bin/cuobjdump"
KERNEL = "trace_photons"
RUNS_FOLDER = os.path.join(engine.HERE, "runs")
INSTRUCTION = re.compile(r"^\s*/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P[0-9T]\s+)?([A-Z][A-Z0-9_]*)")
USER_CONSTANT_BANK = "c[0x3]"

UNITS = [
    ("FP64 units", {"DADD", "DMUL", "DFMA", "DSETP", "DMNMX"}),
    ("FP32 cores", {"FADD", "FMUL", "FFMA", "FSETP", "FMNMX", "FSEL", "FCHK", "FRND", "FSWZADD",
                    "FADD32I", "FMUL32I", "FFMA32I"}),
    ("special function unit (MUFU)", {"MUFU"}),
    ("conversions", {"F2F", "F2I", "I2F", "I2FP", "F2FP", "F2IP"}),
    ("integer units", {"IMAD", "IADD3", "IADD", "IMUL", "LOP3", "LOP", "SHF", "SHL", "SHR", "ISETP", "IMNMX",
                       "LEA", "IABS", "POPC", "FLO", "SEL", "PRMT", "BMSK", "SGXT", "IADD32I", "VIADD", "VIMNMX"}),
    ("atomics (L2)", {"RED", "ATOM", "ATOMG", "ATOMS"}),
    ("constant loads", {"LDC", "ULDC", "LDCU"}),
    ("global memory", {"LDG", "STG"}),
    ("local memory (stack, spills)", {"LDL", "STL"}),
    ("warp-wide", {"SHFL", "VOTE", "VOTEU", "MATCH", "REDUX"}),
    ("moves, special registers", {"MOV", "S2R", "S2UR", "CS2R", "R2UR"}),
    ("control flow", {"BRA", "BSSY", "BSYNC", "EXIT", "CALL", "RET", "WARPSYNC", "BAR", "BREAK", "JMP", "YIELD",
                      "NOP"}),
]


def unit_of(opcode):
    for unit, opcodes in UNITS:
        if opcode in opcodes:
            return unit
    return "uniform datapath" if opcode.startswith("U") else "other"


def kernel_sass(binary):
    sass = subprocess.run([CUOBJDUMP, "-sass", binary], capture_output=True, text=True).stdout
    functions = re.split(r"\n\s*Function : ", sass)
    return next(function for function in functions if KERNEL in function.split("\n", 1)[0])


def instruction_mix(binary):
    opcodes = collections.Counter()
    constant_operands = 0
    for line in kernel_sass(binary).splitlines():
        match = INSTRUCTION.match(line)
        if match:
            opcodes[match.group(1).split(".")[0]] += 1
            constant_operands += line.count(USER_CONSTANT_BANK)
    units = collections.Counter()
    for opcode, count in opcodes.items():
        units[unit_of(opcode)] += count
    return units, opcodes, constant_operands


def report_lines(mixes):
    names = list(mixes)
    all_units = [unit for unit, _ in UNITS] + ["uniform datapath", "other"]
    lines = [f"# Instruction mix of `{KERNEL}` (static: instructions in the code, not how often they run)", "",
             "| Unit | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for unit in all_units:
        lines.append(f"| {unit} | " + " | ".join(str(mixes[name][0][unit]) for name in names) + " |")
    lines.append("| **total** | " + " | ".join(str(sum(mixes[name][0].values())) for name in names) + " |")
    lines.append("| operands read from the Scene (`c[0x3]`) | " +
                 " | ".join(str(mixes[name][2]) for name in names) + " |")
    for name in names:
        top = ", ".join(f"{opcode} {count}" for opcode, count in mixes[name][1].most_common(12))
        lines += ["", f"**{name}, most frequent:** {top}"]
    return lines


def main():
    parser = argparse.ArgumentParser(description="The instruction mix of the GPU kernel, by hardware unit")
    parser.add_argument("--variants", nargs="*", default=[], help="also these build variants, e.g. registers_40")
    args = parser.parse_args()
    mixes = {}
    for precision in engine.PRECISIONS:
        for variant in [None] + args.variants:
            binary = engine.build_engine("gpu", precision, variant=variant)
            mixes[os.path.basename(binary)] = instruction_mix(binary)
    lines = report_lines(mixes)
    folder = os.path.join(RUNS_FOLDER, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "-instruction-mix")
    os.makedirs(folder)
    with open(os.path.join(folder, "report.md"), "w") as file:
        file.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
