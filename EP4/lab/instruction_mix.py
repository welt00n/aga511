import collections
import os
import re
import subprocess

from lab import engine

CUOBJDUMP = engine.cuda_tool("cuobjdump")
KERNEL = "trace_photons"
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


def instruction_count(binary):
    _, opcodes, _ = instruction_mix(binary)
    return {"instructions": sum(opcodes.values()), "local_stores": opcodes.get("STL", 0),
            "local_loads": opcodes.get("LDL", 0)}


def mixes_of_the_kernel(variants=()):
    mixes = {}
    for precision in engine.PRECISIONS:
        for variant in [None, *variants]:
            binary = engine.build_engine("gpu", precision, variant=variant)
            mixes[os.path.basename(binary)] = instruction_mix(binary)
    return mixes


def mixes_as_data(mixes):
    return {name: {"units": dict(units), "opcodes": dict(opcodes), "scene_operands": constant_operands}
            for name, (units, opcodes, constant_operands) in mixes.items()}
