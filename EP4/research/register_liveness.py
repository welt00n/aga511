import collections
import re
import statistics

import common

ADDRESS = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(.*?);")
LIVE = re.compile(r"//\s*\|\s*(\d+)\s*\|")
SOURCE_LINE = re.compile(r'//## File "([^"]+)", line (\d+)(.*)')
INLINED_AT = re.compile(r'inlined at "([^"]+)", line (\d+)')
FUNCTION = re.compile(r"\.text\.(\S+?):?\s*$")
PEAK_MARGIN = 3
SITES_SHOWN = 6
PHYSICS_FILES = ("photon.cuh", "real.cuh", "geometry.cuh")


def kernel_lines(text):
    inside, lines = None, []
    for line in text.splitlines():
        found = FUNCTION.search(line)
        if found:
            inside = common.KERNEL in found.group(1)
        if inside is not False:
            lines.append(line)
    return lines


def live_registers(binary):
    text = common.run([common.tool("nvdisasm"), "-plr", "-lrm", "count", "-c", binary])
    counts = {}
    for line in kernel_lines(text):
        address, live = ADDRESS.search(line), LIVE.search(line)
        if address and live:
            counts[int(address.group(1), 16)] = int(live.group(1))
    return counts


def source_chains(binary):
    text = common.run([common.tool("nvdisasm"), "-g", "-gi", "-c", binary])
    chain, pending, located = None, [], {}
    for line in kernel_lines(text):
        found = SOURCE_LINE.search(line)
        if found:
            pending.append(f"{found.group(1).split('/')[-1]}:{found.group(2)}")
            pending += [f"{path.split('/')[-1]}:{number}" for path, number in INLINED_AT.findall(found.group(3))]
            continue
        address = ADDRESS.search(line)
        if address:
            if pending:
                chain, pending = list(dict.fromkeys(pending)), []
            if chain:
                located[int(address.group(1), 16)] = chain
    return located


def spread(values):
    return f"min {min(values)}, median {statistics.median(values):.0f}, max {max(values)}"


def describe(precision, binary):
    counts, located = live_registers(binary), source_chains(binary)
    peak = max(counts.values())
    print(f"== FP{precision}: {len(counts)} instructions, peak {peak} registers live at once")
    print(f"  where the peak is (live >= {peak - PEAK_MARGIN}; innermost line first, then where it was inlined):")
    sites = collections.Counter(" <- ".join(located.get(address, ["?"])[:3])
                                for address, live in counts.items() if live >= peak - PEAK_MARGIN)
    for site, count in sites.most_common(SITES_SHOWN):
        print(f"    {count:3d} instructions at {site}")
    physics = [live for address, live in counts.items()
               if any(step.startswith(PHYSICS_FILES) for step in located.get(address, []))]
    philox = [live for address, live in counts.items() if located.get(address, ["?"])[0].startswith("curand")]
    print(f"  live inside the photon's physics: {spread(physics)}")
    print(f"  live inside Philox (curand):      {spread(philox)}")
    by_file = collections.defaultdict(list)
    for address, live in counts.items():
        by_file[located.get(address, ["?"])[0].split(":")[0]].append(live)
    print("  by innermost source file: " + "; ".join(f"{name} {spread(values)}" for name, values in
                                                       sorted(by_file.items(), key=lambda item: -max(item[1]))[:5]))


with common.recorded(__file__):
    print("Where do the registers go? nvdisasm counts the registers live at every instruction (-plr -lrm count);")
    print("the line info (-g -gi) maps each instruction to its source line and the lines it was inlined at.")
    print("The compiler needs as many registers as the peak of that count (plus a few reserved), not one per variable.\n")
    folder = common.scratch_folder("liveness")
    for precision in (64, 32):
        describe(precision, common.kernel_cubin(precision, folder))
    print("\nFound on 2026-11-22: FP64 peaks at 74 live (80 allocated), FP32 at 63 (67), both inside sin/cos")
    print("(real.cuh) called for the local azimuth in sample_emission_direction (photon.cuh:56-57), while the")
    print("surface point, Philox's state, the loop's indices and the five 64-bit tallies are all live.")
