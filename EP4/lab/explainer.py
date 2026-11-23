import os

from lab import engine, plan, runs
from lab.report import code, markdown

POINT_FIELDS = {
    "processor, precision, sms, photons, variant, repeat, star": "what was asked (sms empty = all SMs)",
    "SUMMARY.PHOTONS_TIME": "seconds of the photon phase (the kernel, or the CPU loop): the speed is measured on it",
    "SUMMARY.TIME": "seconds of the whole engine run: setup + zero + photons + copy + save",
    "SUMMARY.SETUP_TIME, ZERO_TIME, COPY_TIME, SAVE_TIME": "the serial parts of the run",
    "SUMMARY.PHOTONS_PER_SECOND": "N / PHOTONS_TIME",
    "SUMMARY.RECORDED, BLOCKED, MISSED": "photons that reached an observer's image, hit a spot, or no observer's μ bin",
    "SUMMARY.REGISTERS_PER_THREAD, LOCAL_BYTES_PER_THREAD": "the kernel's registers, and its local memory (spills)",
    "SUMMARY.THREADS_PER_BLOCK, BLOCKS_PER_UNIT, OCCUPANCY": "the launch chosen by the occupancy API",
    "KERNEL[].KERNEL_TIME": "seconds of each pass's kernel, from CUDA events",
    "SMS[].CYCLES": "clock cycles of the slowest SM in each pass: speed with the clock removed",
    "GPU_MONITOR.*": "nvidia-smi sampled every 200 ms: clock (MHz), power (W), temperature, and the share of samples "
                     "with each clock-limit reason active (POWER_CAP, VOLTAGE_CEILING, ...)",
    "IMAGE_DIGEST": "a hash of the observers' images: equal digests = identical images, bit for bit",
    "PROCESS_TIME": "wall-clock seconds of the engine process, measured from Python",
    "validity": "valid (N at or above this GPU's minimum for that many SMs) or INVALID",
}


def number(value, digits=3):
    return f"{value:.{digits}g}" if value is not None else "—"


def times(value):
    return f"{value:.2f}×" if value is not None else "—"


def capability(value):
    return f"{value // 10}.{value % 10}" if isinstance(value, int) else value


def about_one(ratio, tolerance=0.02):
    return ratio is not None and abs(ratio - 1) < tolerance


def measured_star(record):
    config = engine.default_star_config(0, **record["asked"].get("star", {}))
    bytes_per_run = len(config["observers"]) * config["frames"] * config["size"] ** 2 * 4
    return (f"{config['size']}² pixels, {config['frames']} frames, {len(config['observers'])} observers, "
            f"{config['spots']} spots: {bytes_per_run / 1e9:.2f} GB of image per run")


def load_parts(folder, names):
    return {name: runs.read_json(os.path.join(folder, "parts", name + ".json")) for name in names
            if os.path.exists(os.path.join(folder, "parts", name + ".json"))}


# ============================================================ the opening

def opening(record, machine, valid):
    gpu = machine["gpu"] or {}
    banner = "" if valid else ("\n\n> **NOT A VALID MEASUREMENT.** Some runs were below the minimum N (see *What makes "
                               "a valid run*). The tables are kept for testing the report, not for conclusions.")
    minimum_rows = "\n".join(
        f"| {value['binary']} | {value['sms']} | {value['minimum']:.2e} | {value['by_time']:.2e} | "
        f"{value['by_work']:.2e} | {value['resident_threads']} | {record['asked']['photons'][value['binary']]:.2e} |"
        for value in machine["minimums"])
    return [markdown(f"""
# EP4 measurement: {gpu.get('NAME', record['host'])}

{record['started'].replace('T', ' ')} · host `{record['host']}` · engine code `{record['code_hash']}` ·
`{' '.join(record['command'])}`{banner}

**What this is.** The raw results of one `measure` run: every engine run it made is a file in `points/`, every
analysis a file in `parts/`, every figure in `figures/`. The tables below are read from those files when the notebook
runs, so re-running it rebuilds them; the sentences that interpret them were written from the same numbers.

**The star measured:** {measured_star(record)}. Every run in this report traces it, records it and saves it: the
serial part of each run (zeroing, copying back and saving the image) is that of a real high-resolution picture.
"""), markdown(f"""
## The machine

| | |
|---|---|
| GPU | {gpu.get('NAME', '—')}, compute capability {capability(gpu.get('COMPUTE_CAPABILITY', '—'))}, {gpu.get('SMS', '—')} SMs |
| per SM | {gpu.get('REGISTERS_PER_SM', '—')} registers, {gpu.get('MAX_THREADS_PER_SM', '—')} threads, {gpu.get('MAX_BLOCKS_PER_SM', '—')} blocks |
| CPU (correctness only) | {machine.get('cpu', '—')}: the sequential engine the GPU's images are checked against; its speed is not compared |
| software | {machine.get('nvcc', '—')}; driver {machine.get('driver', '—')} |
"""), markdown(f"""
## What makes a valid run

{plan.MINIMUM_RULE}

| binary | SMs | minimum N | by time (1 s) | by work (32 chunks per thread) | resident threads | N used here |
|---|---|---|---|---|---|---|
{minimum_rows}

On one SM the work rule asks for far less (one SM's threads), and the same N simply runs longer. **Every timed run is
above the minimum**: `measure` refuses to run otherwise. The only runs at a smaller N are the validation checks
against the sequential CPU (N = 10⁸): they compare images, not times, and the CPU side alone would need about an hour at 10¹⁰.
""")]


# ============================================================ 1. validation

def validation(data):
    found = {f"{check['name']}-{check['processor']}{check['precision']}": check for check in data["checks"]}
    independent, precision = found["independent-gpu64"]["data"], found["precision-gpu64"]["data"]
    gpu64, gpu32 = found["gpu-gpu64"]["data"], found["gpu-gpu32"]["data"]
    ratio = precision["noise"] / precision["moved"] if precision["moved"] else float("inf")
    return [markdown("""
## 1. Is the physics right? (validation)

**Question.** Does the GPU compute the same star as the sequential CPU engine, both photon by photon and in the
statistics of the image?

**Method.**
- *GPU vs CPU, the same photons:* the same seed, so the same random numbers; the images compared bit for bit.
- *GPU vs CPU, independent photons:* the same seed, so the same star and spots, but different chunks (each chunk is
  its own random stream): two independent Monte Carlo images of the same star. The χ² of their difference (frames
  summed), per degree of freedom: the same physics gives χ²/dof ≈ 1 (|σ| < 3 passes). Not different seeds: the seed
  also places the spots, so another seed is another star.
- *FP32 vs FP64:* the same seed, so the same photons; the fraction of photons that land in another pixel in FP32,
  next to the Poisson noise per lit pixel.
"""), code("""
checks = load("validation")["checks"]
table(checks, [("check", lambda c: f"{c['name']} ({c['processor']}{c['precision']})", None),
               ("N", lambda c: c["data"].get("photons"), ".1e"),
               ("passed", "passed", lambda v: {True: "yes", False: "NO"}.get(v, "—")),
               ("result", "headline", None)])
"""), markdown(f"""
**What it means.**
- GPU vs the sequential CPU with independent photons: χ²/dof = {independent['chi2_per_dof']:.4f} over
  {independent['dof']} blocks of {independent.get('block_pixels', 1)}×{independent.get('block_pixels', 1)} pixels
  (summed so that each block holds enough photons for the test), {independent['sigma_from_1']:+.2f} σ from 1:
  {'the same physics within the noise' if abs(independent['sigma_from_1']) < 3 else 'the images DIFFER beyond the noise'}.
- FP32 moves {precision['moved']:.2e} of the photons to another pixel; the noise per lit pixel is
  {precision['noise']:.2e}, so the FP32 error is {ratio:.0f}× smaller than the noise. FP32 is
  {precision['speed_ratio']:.1f}× faster than FP64 on this GPU.
- GPU vs CPU, FP64: {'identical, bit for bit' if gpu64['identical'] else f"{gpu64['differing_pixels']} pixels differ"}.
  FP32: {'identical' if gpu32['identical'] else f"{gpu32['moved']:.2e} of the photons moved"} (the GPU's `sinf`,
  `cosf` and fused multiply-adds round differently from the CPU's).

**Caveats.** The χ² test is statistical: about 1 run in 370 fails it by chance. The bit-for-bit results hold for this
build and this seed; they don't depend on N.
""")]


# ============================================================ 2. speed

def speed(data):
    every, ratio = data["all_sms"], data["all_sms_over_one_sm"]
    rows = {(row["precision"], row["sms"] == 1): row for row in data["rows"]}
    efficiency = {p: ratio[f"fp{p}"] / every for p in (64, 32)}
    return [markdown(f"""
## 2. How fast is the GPU, one SM and all of them? (speed)

**Question.** Photons per second on one SM and on all {every}, in FP64 and FP32. The unit of comparison is the SM:
the GPU against itself, not against another processor.

**Method.** The measured star (its image is in the opening, under *The machine*), the same N on one SM and on all of
them, repeated. The speed is N over the photon phase alone (the kernel): setup and save are left out. The one-SM run
is also the first point of the strong sweep (section 5).
"""), code("""
speed = load("speed")
table(speed["rows"], [("precision", lambda r: f"FP{r['precision']}", None), ("SMs", "sms", None), ("N", "photons", ".1e"),
                      ("photons/s (median)", lambda r: r["photons_per_second"]["median"], ".3g"),
                      ("min – max", lambda r: f"{r['photons_per_second']['min']:.3g} – {r['photons_per_second']['max']:.3g}", None),
                      ("SM clock (MHz)", "sm_clock_mhz", ".0f"), ("power (W)", "power_w", ".0f"),
                      ("power cap", "power_cap_share", ".0%"), ("repeats", lambda r: r["photons_per_second"]["repeats"], None)])
figure("figures/speed.png")
"""), markdown(f"""
**What it means.**
- One SM: {rows[(64, True)]['photons_per_second']['median']:.3g} photons/s in FP64,
  {rows[(32, True)]['photons_per_second']['median']:.3g} in FP32.
- All {every} SMs: {rows[(64, False)]['photons_per_second']['median']:.3g} in FP64 ({ratio['fp64']:.1f}× one SM, so each
  SM keeps {efficiency[64]:.0%} of its one-SM speed), {rows[(32, False)]['photons_per_second']['median']:.3g} in FP32
  ({ratio['fp32']:.1f}×, {efficiency[32]:.0%}).
- FP32 over FP64: {data['fp32_over_fp64']['one_sm']:.1f}× on one SM, {data['fp32_over_fp64']['all_sms']:.1f}× on all
  of them. This GPU has few FP64 units per SM, so an FP64 photon costs far more than an FP32 one.

**Caveats.** With every SM busy in FP32 the GPU reaches its power limit and lowers its clock, so the speed per SM falls
(the SM clock column); section 5 shows the same scaling in cycles, without the clock.
""")]


# ============================================================ 3. kernel

def scene_sentence(by_name):
    ratios = [by_name[p]["scene_in_global"]["time_vs_baseline"] for p in ("fp64", "fp32")]
    measured = f"{times(ratios[0])} the time in FP64, {times(ratios[1])} in FP32"
    if all(about_one(ratio, 0.03) for ratio in ratios):
        return (f"Constant vs global memory for the scene: no difference ({measured}). Every thread of a warp reads the "
                "same scene value at the same time, which the constant cache broadcasts, and L1 serves just as well.")
    return f"Constant vs global memory for the scene: {measured}."


def fma_sentence(by_name):
    fp64 = by_name["fp64"]["no_fma"]
    measured = (f"No FMA (separate multiply and add): {times(fp64['time_vs_baseline'])} the time in FP64, "
                f"{times(by_name['fp32']['no_fma']['time_vs_baseline'])} in FP32")
    if fp64["image_equals_baseline"]:
        return measured + "; the FP64 image is the same: fusing changed no photon's pixel here."
    return measured + "; the FP64 image changes, because a fused multiply-add rounds once instead of twice."


def sections_sentence(by_name, top):
    ratios = [by_name[p]["sections"]["time_vs_baseline"] for p in ("fp64", "fp32")]
    cost = (f"The section timers cost {times(ratios[0])} (FP64) and {times(ratios[1])} (FP32)"
            + (": almost nothing." if all(about_one(r) for r in ratios) else ": measuring changes what is measured."))
    return (cost + f" The largest section is *{top['fp64']['name']}* ({top['fp64']['share']:.0%} of the cycles, FP64) "
                   f"and *{top['fp32']['name']}* ({top['fp32']['share']:.0%}, FP32).")


def kernel(data):
    by_name = {precision: {row["variant"]: row for row in data["variants"][precision]} for precision in ("fp64", "fp32")}
    top = {precision: max(data["sections"][precision], key=lambda s: s["share"]) for precision in ("fp64", "fp32")}
    return [markdown("""
## 3. What the kernel is made of (kernel)

**Question.** What does the compiler make of `trace_photons`, and which changes to it matter?

**Method.** The resources come from the compiler (`ptxas -v`) and the CUDA runtime (`cudaFuncGetAttributes`, the
occupancy API). The instruction mix counts the instructions in the kernel's machine code (SASS) by the unit that runs
them. Each variant is a rebuild with one change, run at the same N as the baseline:
- `scene_in_global`: the scene in global memory instead of constant memory;
- `no_fma`: no fused multiply-adds (`--fmad=false`);
- `sections`: timers around each piece of the photon (cycles per section), to see where the time goes.
"""), code("""
kernel = load("kernel")
print("Resources")
table([dict(r, precision=p) for p, r in kernel["resources"].items()],
      [("precision", "precision", None), ("registers", "registers", None), ("spills (B)", "spill_bytes", None),
       ("constant (B)", "constant_bytes", None), ("threads/block", "threads_per_block", None),
       ("blocks/SM", "blocks_per_sm", None), ("occupancy", "occupancy", ".2f")])
mix = kernel["instruction_mix"]
units = sorted({unit for counts in mix.values() for unit in counts})
print("Instruction mix (static: instructions in the code, not how often they run)")
table([dict({"unit": unit}, **{name: counts.get(unit, 0) for name, counts in mix.items()}) for unit in units],
      [("unit", "unit", None)] + [(name, name, None) for name in mix])
for precision, rows in kernel["variants"].items():
    print(f"Variants, {precision}")
    table(rows, [("variant", "variant", None), ("registers", "registers", None), ("spills (B)", "spill_bytes", None),
                 ("kernel (s)", lambda r: r["kernel_time"]["median"], ".3f"),
                 ("time vs baseline", "time_vs_baseline", ".3f"), ("cycles vs baseline", "cycles_vs_baseline", ".3f"),
                 ("same image", "image_equals_baseline", lambda v: "yes" if v else "no")])
for precision, sections in kernel["sections"].items():
    print(f"Where the time goes, {precision} (cycles per section, the sections build)")
    table(sorted(sections, key=lambda s: -s["share"]), [("section", "name", None), ("share", "share", ".1%")])
"""), markdown(f"""
**What it means.**
- Registers: {data['resources']['fp64']['registers']} (FP64) and {data['resources']['fp32']['registers']} (FP32),
  spills {data['resources']['fp64']['spill_bytes']} / {data['resources']['fp32']['spill_bytes']} bytes; the launch
  gives occupancy {data['resources']['fp64']['occupancy']:.2f} / {data['resources']['fp32']['occupancy']:.2f}
  (section 4 explains why).
- {scene_sentence(by_name)}
- {fma_sentence(by_name)}
- {sections_sentence(by_name, top)}

**Caveats.** The mix is static: a `NOP` or an FP64 instruction in a branch that never runs is counted like any other
(`research/nop_runs.py`, `research/fp64_in_fp32_kernel.py`). Times are medians of the repeats; their spread is in
`parts/kernel.json`.
""")]


# ============================================================ 4. occupancy

def best_and_full(rows):
    best = min(rows, key=lambda row: row["cycles_vs_baseline"] or 9)
    full = [row for row in rows if row["occupancy"] >= 0.99]
    return best, full[0] if full else None


def occupancy(data):
    limits, here = data["device"], data["this_gpu"]
    meaning = []
    for precision in ("fp64", "fp32"):
        best, full = best_and_full(data["rows"][precision])
        launch = data["rows"][precision][0]["occupancy"]
        line = (f"- {precision.upper()}: the uncapped build uses {here[precision]['registers']} registers, so the "
                f"registers allow occupancy {here[precision]['occupancy_at_most']:.2f}"
                + (f"; the launch gets {launch:.2f}, because blocks fit only whole" if launch < here[precision][
                    "occupancy_at_most"] - 0.005 else "")
                + f". The fewest cycles: {best['registers']} registers ({best['cycles_vs_baseline']:.3f}× the "
                  f"uncapped cycles, occupancy {best['occupancy']:.2f}).")
        if full:
            line += (f" Full occupancy needs {full['registers']} registers: {full['cycles_vs_baseline']:.3f}× the "
                     f"cycles, with {full['spill_bytes']} bytes spilled to local memory.")
        meaning.append(line)
    targets = "\n".join(
        f"- {target['device']['NAME']} ({target['target']}): FP64 {target['precisions']['fp64']['registers']} registers "
        f"→ occupancy at most {target['precisions']['fp64']['occupancy_at_most']:.2f}; FP32 "
        f"{target['precisions']['fp32']['registers']} → {target['precisions']['fp32']['occupancy_at_most']:.2f} "
        "(compiled here, not run)." for target in data["targets"]) or "- (no other target asked)"
    return [markdown(f"""
## 4. Registers and occupancy (occupancy)

**Question.** The registers each thread needs limit how many warps an SM can hold (the occupancy). Would more warps
per SM make the kernel faster?

**The arithmetic.** An SM has {limits['REGISTERS_PER_SM']} registers and holds at most
{limits['MAX_THREADS_PER_SM']} threads ({limits['MAX_THREADS_PER_SM'] // limits['WARP_SIZE']} warps of
{limits['WARP_SIZE']}). Registers are given per warp, in steps of 8 per thread: a warp of threads needing *r* registers
takes ⌈r/8⌉·8·32. So the warps that fit are {limits['REGISTERS_PER_SM']} / (⌈r/8⌉·8·32), at most
{limits['MAX_THREADS_PER_SM'] // limits['WARP_SIZE']}, and the occupancy is that over
{limits['MAX_THREADS_PER_SM'] // limits['WARP_SIZE']}.

**Method.** Rebuild the kernel with `-maxrregcount` caps; the compiler then keeps fewer values in registers and spills
the rest to local memory. Each cap runs at the same N, the caps interleaved (forward, then backward) so that a clock
drifting during the sweep hits all of them alike. The speed is compared in **cycles of the slowest SM**, which don't
depend on the clock.
"""), code("""
occupancy = load("occupancy")
print("This GPU, uncapped")
table([dict(r, precision=p) for p, r in occupancy["this_gpu"].items()],
      [("precision", "precision", None), ("registers", "registers", None), ("per warp", "registers_per_warp", None),
       ("warps the registers allow", "warps_allowed_by_registers", None), ("of", "max_warps", None),
       ("threads per SM", "threads_per_sm", None), ("occupancy at most", "occupancy_at_most", ".2f")])
for precision, rows in occupancy["rows"].items():
    print(f"The caps, {precision}")
    table(rows, [("cap", "cap", None), ("registers", "registers", None), ("warps allowed", "warps_allowed_by_registers", None),
                 ("occupancy", "occupancy", ".2f"), ("spills (B)", "spill_bytes", None),
                 ("cycles vs uncapped", "cycles_vs_baseline", ".3f"), ("time vs uncapped", "time_vs_baseline", ".3f"),
                 ("MHz", "sm_clock_mhz", ".0f"), ("power cap", "power_cap_share", ".0%"),
                 ("same image", "image_equals_baseline", lambda v: "yes" if v else "no")])
figure(occupancy["figure"])
"""), markdown("**What it means.**\n" + "\n".join(meaning) + f"""

Other GPUs, from the compiler alone:
{targets}

**Caveats.** "Power cap" is the share of `nvidia-smi` samples in which the driver held the clock down for power; the
cycles are immune to it, the times are not. On a GPU where the kernel waits on memory or on long latencies, more warps
can hide that wait; where it is bound by its arithmetic units, they can't.
""")]


# ============================================================ 5. scaling

def strong_rows_code(label):
    return f"""
strong = load("scaling")["{label}"]
table([{{"sms": n, "photons": strong["photon_time"][str(n)], "whole": strong["whole_run_time"][str(n)],
         "s_photons": strong["speedup_photon_phase"][str(n)], "s_whole": strong["speedup_whole_run"][str(n)],
         "s_cycles": strong["speedup_cycles"].get(str(n)), "mhz": strong["sm_clock_mhz"].get(str(n))}}
       for n in strong["sms"]],
      [("SMs", "sms", None), ("photon phase (s)", "photons", ".3f"), ("whole run (s)", "whole", ".3f"),
       ("speedup, photons", "s_photons", ".2f"), ("speedup, whole run", "s_whole", ".2f"),
       ("speedup, cycles", "s_cycles", ".2f"), ("SM clock (MHz)", "mhz", ".0f")])
"""


def strong_meaning(strong, precision):
    every = str(max(strong["sms"]))
    photon, whole = strong["amdahl_photon_phase"], strong["amdahl_whole_run"]
    cycles = strong["speedup_cycles"].get(every)
    return (f"- FP{precision}, N = {strong['photons']:.2e}: {strong['speedup_photon_phase'][every]:.1f}× on {every} SMs "
            f"for the photon phase (Amdahl p = {photon['p']:.4f}), {strong['speedup_whole_run'][every]:.1f}× for the "
            f"whole run (p = {whole['p']:.4f}, ceiling 1/s = {whole['ceiling']:.0f}); in cycles "
            f"{number(cycles)}×, so the difference from the seconds is the clock. The serial share measured from the "
            f"phases: {strong['serial_fraction_measured']:.4f} (fitted: {whole['s']:.4f}).")


def ceiling_sentence(strong, precision):
    every = str(max(strong["sms"]))
    fit = strong["amdahl_whole_run"]
    serial = strong["whole_run_time"][every] - strong["photon_time"][every]
    return (f"- FP{precision}: on {every} SMs a run is {strong['photon_time'][every]:.1f} s of photons and "
            f"{serial:.1f} s of everything else (setup, zeroing, copying back and saving the image), so the whole run "
            f"is {strong['speedup_whole_run'][every]:.1f}× faster than on one SM while the photons are "
            f"{strong['speedup_photon_phase'][every]:.1f}×. Amdahl: s = {fit['s']:.4f}, so the whole run can never "
            f"pass 1/s = {fit['ceiling']:.0f}×, however many SMs.")


def scaling_section(data):
    size = data["problem-size-fp32"]
    sizes = [str(n) for n in size["sizes"]]
    weak_lines = "\n".join(
        f"- FP{p}: on {max(data[f'weak-fp{p}']['sms'])} SMs the efficiency is "
        f"{data[f'weak-fp{p}']['efficiency_photon_phase'][str(max(data[f'weak-fp{p}']['sms']))]:.3f}; Gustafson's "
        f"serial share fitted s′ = {data[f'weak-fp{p}']['gustafson_whole_run']['s_prime']:.4f}, measured "
        f"{data[f'weak-fp{p}']['serial_fraction_measured_on_all_sms']:.4f}." for p in (64, 32))
    return [markdown("""
## 5. Scaling: Amdahl and Gustafson (scaling)

**Question.** How does the time change with the number of SMs, for a fixed problem (strong scaling, Amdahl) and for a
problem that grows with the SMs (weak scaling, Gustafson)?

**Method.** The engine's `--sms n` keeps the work on n SMs. Strong: the same N on every n. Weak: N proportional to n.
Each point is repeated, interleaved; medians are used. Two times are kept: the **photon phase** (the kernel) and the
**whole run** (with the setup, zeroing and saving, which don't get faster with more SMs). Amdahl's law
T(n) = T(1)·(s + p/n) is fitted by least squares; Gustafson's scaled speedup is n − s′(n − 1).

### 5.1 Strong scaling
"""), code(strong_rows_code("strong-fp64") + '\nfigure("figures/strong_fp64.png")'),
        code(strong_rows_code("strong-fp32") + '\nfigure("figures/strong_fp32.png")'),
        markdown("**What it means.**\n" + strong_meaning(data["strong-fp64"], 64) + "\n" +
                 strong_meaning(data["strong-fp32"], 32)),
        markdown("### 5.2 Weak scaling"),
        code("""
for precision in (64, 32):
    weak = load("scaling")[f"weak-fp{precision}"]
    print(f"FP{precision}")
    table([{"sms": n, "photons": weak["photons"][str(n)], "time": weak["photon_time"][str(n)],
            "whole": weak["whole_run_time"][str(n)], "scaled": weak["scaled_speedup_whole_run"][str(n)],
            "efficiency": weak["efficiency_photon_phase"][str(n)]} for n in weak["sms"]],
          [("SMs", "sms", None), ("N", "photons", ".2e"), ("photon phase (s)", "time", ".3f"),
           ("whole run (s)", "whole", ".3f"), ("scaled speedup", "scaled", ".2f"), ("efficiency", "efficiency", ".3f")])
    figure(f"figures/weak_fp{precision}.png")
    figure(f"figures/strong_vs_weak_fp{precision}.png")
"""), markdown("**What it means.**\n" + weak_lines),
        markdown("### 5.3 Amdahl's ceiling, at valid N: the image is the serial part"),
        code('figure("figures/amdahl_ceiling_fp64.png")\nfigure("figures/amdahl_ceiling_fp32.png")'),
        markdown("**What it means.**\n" + ceiling_sentence(data["strong-fp64"], 64) + "\n" +
                 ceiling_sentence(data["strong-fp32"], 32) + """
- The photons are the parallel part; the image is the serial part: one disk saves it, one PCIe link copies it, and
  that time doesn't shrink with the SMs. The bigger the image, the lower the ceiling.
"""),
        markdown("### 5.4 p(N): the parallel fraction grows with the problem (valid N only)"),
        code("""
size = load("scaling")["problem-size-fp32"]
table([{"n": int(n), "p_photons": size["p_photon_phase"][str(n)], "p_whole": size["p_whole_run"][str(n)]}
       for n in size["sizes"]], [("N", "n", ".0e"), ("p, photon phase", "p_photons", ".5f"), ("p, whole run", "p_whole", ".5f")])
figure("figures/problem_size.png")
"""),
        markdown(f"""
**What it means.** Amdahl's p is not a property of the program alone: from N = {float(sizes[0]):.0e} to
{float(sizes[-1]):.0e} it goes from {size['p_photon_phase'][sizes[0]]:.3f} to {size['p_photon_phase'][sizes[-1]]:.5f}
for the photon phase, and from {size['p_whole_run'][sizes[0]]:.3f} to {size['p_whole_run'][sizes[-1]]:.4f} for the
whole run. The work grows with N; the serial part (the image) doesn't, so the more photons, the larger the share
that the SMs can share.

**Caveats.** On many SMs in FP32 the clock falls under the power cap, so seconds understate the scaling; the cycles
don't (the "speedup, cycles" column). Every run here is above the minimum N: none is too short to be a measurement.
""")]


# ============================================================ 6. costs

def cost_row(data, size, frames, observers):
    return next(row for row in data["rows"] if (row["size"], row["frames"], row["observers"]) == (size, frames, observers))


def cost_sentence(row):
    time = plan.duration(row["seconds"]) if row["seconds"] else "—"
    passes = f"{row['passes']} pass" + ("es" if row["passes"] and row["passes"] > 1 else "") if row["passes"] \
        else "does not fit"
    return (f"{row['size']}², {row['frames']} frames (nμ {row['nmu']}), {row['observers']} observers: "
            f"N = {row['photons']:.1e}, {time} in FP32, {row['bytes'] / 1e9:.1f} GB ({passes}); blur "
            f"{row['rotation_blur_px']:.0f} px (rotation, equator), {row['pole_blur_px']:.0f} px (polar bin)")


def large_n_row(rows, precision, photons):
    return next(row for row in rows if row["precision"] == precision and row["photons"] == photons)


def large_n_cells(data):
    found = data.get("large_n")
    if not found:
        return []
    rows, fp32, fp64 = found["rows"], found["models"]["fp32"], found["models"]["fp64"]
    top32, mid32, mid64 = large_n_row(rows, 32, 10 ** 13), large_n_row(rows, 32, 10 ** 12), large_n_row(rows, 64, 10 ** 11)
    return [markdown(f"""
### 6.1 How large an N for the best scaling, and what it costs to measure

**Question.** The whole run (photons + saving the image) scales less than the photons, because the image's serial part
is fixed. How large must N be for the whole run to come close to ideal, and how long would that take to measure?

**Method.** From this measurement: the speed on one SM and on all of them, and the serial part of a run
({fp32['serial_seconds']:.1f} s in FP32, {fp64['serial_seconds']:.1f} s in FP64: setup, copying back and saving the
image). Then T(n) = N / speed(n) + serial, for any N. A strong sweep needs the same N on every SM count, so its cost
is dominated by the run on one SM.
"""), code("""
large = load("costs")["large_n"]
def hours(seconds):
    return f"{seconds:.0f} s" if seconds < 90 else f"{seconds / 60:.0f} min" if seconds < 5400 else f"{seconds / 3600:.1f} h"
table(large["rows"], [("precision", lambda r: f"FP{r['precision']}", None), ("N", "photons", ".0e"),
                      ("1 SM", lambda r: hours(r["one_sm_seconds"]), None),
                      ("all SMs", lambda r: hours(r["all_sms_seconds"]), None),
                      ("whole-run speedup", "whole_run_speedup", ".1f"),
                      ("strong sweep (1 repeat)", lambda r: hours(r["strong_sweep_seconds"]), None),
                      ("weak sweep (1 repeat)", lambda r: hours(r["weak_sweep_seconds"]), None)])
figure("figures/costs_large_n.png")
"""), markdown(f"""
**What it means.**
- The measured points sit on the predicted curve: the whole run's speedup is the photons' speedup diluted by the fixed
  serial part, nothing else.
- FP32 at N = 10¹²: {mid32['whole_run_speedup']:.0f}× on the whole run, and a strong sweep of
  {plan.duration(mid32['strong_sweep_seconds'])} (one repeat). At 10¹³: {top32['whole_run_speedup']:.1f}×, but the run
  on one SM alone takes {plan.duration(top32['one_sm_seconds'])}.
- FP64 gets there sooner, because each photon costs more: at 10¹¹ it is already {mid64['whole_run_speedup']:.0f}×
  ({plan.duration(mid64['strong_sweep_seconds'])} for the sweep).
- The cheap way to show near-ideal scaling at large N is the weak sweep: every point runs about as long as the run on
  all SMs.
""")]


def costs(data):
    showcase, wide = cost_row(data, 1024, 300, 4), cost_row(data, 1024, 300, 7)
    big, sharp_big = cost_row(data, 2048, 300, 4), cost_row(data, 2048, 600, 4)
    return [markdown(f"""
## 6. Resolution and image size: what each one costs (costs)

**Question.** What does a sharper picture cost, and what does a bigger one cost? They are two different axes.

**Axis 1, resolution: the bins, and so the computation.** A photon leaves the star in one direction; the direction
picks its frame (one of `frames` bins of the rotation) and its μ bin (one of nμ bins of μ = cos i). A frame of one
observer collects every photon of its cell, each projected along its own direction, so it **blends the views inside
the cell**: a blur of about R·2π/frames pixels along the rotation, and R·Δi across it, where Δi ≈ (2/nμ)/sin i is the
cell's width in inclination (R = size/2 pixels). Bins equal in μ are equal in photons but not in angle: Δi is
smallest at the equator and largest at the poles (the polar bin spans 0 to ≈ 2/√nμ radians). With nμ = frames/π the
two blurs match at the equator. Finer bins share the photons among more cells, so the photons needed for an SNR are

N = SNR² · nμ · frames · (π/4)·size² · (1 + 2a/3), and the time is passes × N / (photons per second).

**Axis 2, image size: the bytes, and so the memory.** One observer's image is frames × size² counts of 4 bytes; a run
holds all its observers: observers × frames × size² × 4 bytes. When they don't fit in the GPU's memory, the run makes
**passes**, and each pass traces all N photons again: memory turns into time.

**Which knob is on which axis.** The **observers** are only on the memory axis: each one keeps the photons of its
own bin, whatever the others do, so more observers need no more photons, only more memory (and passes). The
**frames** are on both: they sharpen the rotation, cost photons, and multiply the bytes. The **size** is on both:
more pixels need more photons for the same SNR, and more bytes. nμ only costs photons.

**Method.** Not a run: the formulas above with this GPU's measured speed ({data['speed_fp32']:.3g} photons/s in FP32,
{data['speed_fp64']:.3g} in FP64; {data['speed_source']}) and its memory for the image
({data['memory_for_image'] / 1e9:.2f} GB), for SNR {data['snr']} at the limb. The SNR formula is checked by every
`make` (measured = predicted), and the passes by the engine's `--describe`.
"""), code("""
costs = load("costs")
def hours(seconds):
    return "—" if seconds is None else f"{seconds / 60:.1f} min" if seconds < 5400 else f"{seconds / 3600:.1f} h"
table(costs["rows"], [("size", lambda r: f"{r['size']}²", None), ("frames", "frames", None), ("nμ", "nmu", None),
                      ("observers", "observers", None), ("N for the SNR", "photons", ".1e"),
                      ("time FP32", lambda r: hours(r["seconds"]), None),
                      ("time FP64", lambda r: hours(r["seconds_fp64"]), None),
                      ("GB", lambda r: r["bytes"] / 1e9, ".2f"), ("passes", "passes", None),
                      ("blur: rotation (px)", "rotation_blur_px", ".1f"), ("blur: polar bin (px)", "pole_blur_px", ".0f"),
                      ("video (s at 30 fps)", "video_seconds_at_30fps", ".0f")])
figure("figures/costs_compute.png")
figure("figures/costs_memory.png")
"""), markdown(f"""
**What it means.**
- The showcase: {cost_sentence(showcase)}.
- The same with 7 observers: {cost_sentence(wide)}. More observers, the same N: only the memory grew, and with it the
  passes, which each trace all N again.
- Twice the size: {cost_sentence(big)}. Four times the pixels: 4× the photons and 4× the bytes, and the blur doubles in
  pixels (it is the same in angle).
- Twice the size **and** the same blur in pixels: {cost_sentence(sharp_big)}. Twice the frames and twice nμ on top of
  4× the pixels: about 16× the photons of the 1k showcase. Sharpness at a fixed pixel scale grows as size⁴.
- A picture as sharp as its pixels (a blur of 1 px) would need frames ≈ π·size and nμ ≈ size: at 1k about 3200 frames
  and 1019 bins, ~10¹⁴–10¹⁵ photons: out of reach with this algorithm. The honest description of an image is its
  pixels **and** its blur.

""")] + large_n_cells(data) + [markdown("""
**Caveats.** The times are predictions from the measured speed (`make` prints the same prediction and then measures
it). The polar bin is always the blurriest with equal-μ bins; observers away from the exact poles are sharper. An
algorithm that adds every photon to every observer and frame with its exact geometry ("peel-off") would remove the
blur and the 1/(nμ·frames) dilution, but it is not EP2's algorithm.
""")]


# ============================================================ the appendix

def appendix(parts):
    fields = "\n".join(f"| `{name}` | {meaning} |" for name, meaning in POINT_FIELDS.items())
    files = "\n".join(f"- `parts/{name}.json`" for name in parts)
    return [markdown(f"""
## Appendix: the data

**The folder.**
- `run.json`: what was asked (the options, the reference N per binary), the machine, the status.
- `machine.json`: the GPU's limits, the software, the environment checks, the minimum N.
- `plan.json`: every planned run with its prediction (seconds, memory, passes) and its validity.
- `points/<run>.json`: one file per engine run (the name says the binary, the variant, the SMs, N and the repeat).
- `parts/<part>.json`: the analyses, which these tables read:
{files}
- `figures/`: the figures, each with a `.json` of the numbers it plots.
- `progress.log`: each run's measured vs predicted time.

**The fields of a point.**

| field | meaning |
|---|---|
{fields}
""")]


SECTIONS = {"costs": costs, "validation": validation, "speed": speed, "kernel": kernel, "occupancy": occupancy,
            "scaling": scaling_section}
ORDER = ("validation", "speed", "kernel", "occupancy", "scaling", "costs")


def measurement_cells(folder, record, machine, valid):
    parts = load_parts(folder, ORDER)
    cells = opening(record, machine, valid)
    for name in ORDER:
        if name in parts:
            cells += SECTIONS[name](parts[name])
    return cells + appendix(parts)
