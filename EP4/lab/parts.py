import os
import statistics
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lab import budget, checks, device, engine, instruction_mix, plan, scaling, speeds  # noqa: E402

PARTS = ("validation", "speed", "kernel", "occupancy", "scaling", "costs")
COST_SIZES = (512, 1024, 2048)
COST_FRAMES = (150, 300, 600, 1000)
COST_OBSERVERS = (4, 7)
COST_SNR = 10
KERNEL_VARIANTS = ("scene_in_global", "no_fma", "sections")
REGISTER_CAPS = {64: [None, 72, 64, 56, 48, 40, 32], 32: [None, 64, 56, 48, 40, 32]}
SM_STEPS = (1, 2, 3, 4, 6, 9, 12, 18, 24, 36, 48)
PROBLEM_FACTORS = (0.4, 1, 2)
DIGEST_PHOTONS = 2 * 10 ** 10
VALIDATION_PHOTONS = 10 ** 8
COLORS = {64: scaling.PHOTON_PHASE, 32: scaling.WHOLE_RUN}

for caps in REGISTER_CAPS.values():
    for cap in caps:
        if cap:
            engine.GPU_VARIANTS[f"registers_{cap}"] = [f"-maxrregcount={cap}"]


# ============================================================ what a measurement asks for

def all_sms():
    return plan.described("gpu", 64, None, None, "{}")["COMPUTE_UNITS_TOTAL"]


def sm_counts():
    total = all_sms()
    return [n for n in SM_STEPS if n < total] + [total]


def on(sms):
    return None if sms == all_sms() else sms


def reference_photons(seconds):
    return {f"gpu{precision}": plan.reference_photons("gpu", precision, seconds) for precision in engine.PRECISIONS}


def reference(settings, processor, precision):
    return settings["photons"][f"{processor}{precision}"]


def repeats(settings):
    return range(1, settings["repeats"] + 1)


def star(settings):
    return settings.get("star", {})


def measured(settings, processor, precision, photons, sms=None, repeat=1):
    return plan.point(processor, precision, photons, sms, repeat=repeat, **star(settings))


def baseline(settings, precision, repeat):
    return measured(settings, "gpu", precision, reference(settings, "gpu", precision), repeat=repeat)


def add_speed(the_plan, settings):
    for repeat in repeats(settings):
        for precision in engine.PRECISIONS:
            the_plan.add(baseline(settings, precision, repeat), "speed")
            the_plan.add(dict(baseline(settings, precision, repeat), sms=1), "speed")


def add_kernel(the_plan, settings):
    for repeat in repeats(settings):
        for precision in engine.PRECISIONS:
            the_plan.add(baseline(settings, precision, repeat), "kernel")
            for variant in KERNEL_VARIANTS:
                the_plan.add(dict(baseline(settings, precision, repeat), variant=variant), "kernel")


def register_variant(cap):
    return f"registers_{cap}" if cap else None


def add_occupancy(the_plan, settings):
    for repeat in repeats(settings):
        for precision, caps in REGISTER_CAPS.items():
            for cap in caps if repeat % 2 else caps[::-1]:
                the_plan.add(dict(baseline(settings, precision, repeat), variant=register_variant(cap)), "occupancy")


def add_scaling(the_plan, settings):
    for repeat in repeats(settings):
        for precision in engine.PRECISIONS:
            photons = reference(settings, "gpu", precision)
            for sms in sm_counts():
                the_plan.add(measured(settings, "gpu", precision, photons, on(sms), repeat), f"strong-fp{precision}")
                the_plan.add(measured(settings, "gpu", precision, photons * sms // all_sms(), on(sms), repeat),
                             f"weak-fp{precision}")
        for photons in problem_sizes(settings):
            for sms in (1, all_sms()):
                the_plan.add(measured(settings, "gpu", 32, photons, on(sms), repeat), "problem-size-fp32")


def problem_sizes(settings):
    return [plan.nice_above(factor * reference(settings, "gpu", 32)) for factor in PROBLEM_FACTORS]


def validation_checks(settings):
    gpu64 = reference(settings, "gpu", 64)
    return [("independent", "gpu", 64, VALIDATION_PHOTONS), ("precision", "gpu", 64, gpu64),
            ("gpu", "gpu", 64, VALIDATION_PHOTONS), ("gpu", "gpu", 32, VALIDATION_PHOTONS)]


def check_seconds(name, processor, precision, photons):
    gpu = photons / speeds.speed_of("gpu", precision)[0]
    if name == "precision":
        return photons / speeds.speed_of("gpu", 64)[0] + photons / speeds.speed_of("gpu", 32)[0] + 2
    return photons / speeds.speed_of("cpu", precision)[0] + gpu + 2


def add_validation(the_plan, settings):
    for precision in engine.PRECISIONS:
        the_plan.add(measured(settings, "gpu", precision, DIGEST_PHOTONS), "digests")
    for name, processor, precision, photons in validation_checks(settings):
        the_plan.add_task("validation", f"{name}-{processor}{precision}", check_seconds(name, processor, precision,
                                                                                         photons),
                          f"{checks.CHECKS[name].__name__} at N = {photons:.1e} (N set by the statistics)")


def add_costs(the_plan, settings):
    return None


ADD = {"costs": add_costs, "validation": add_validation, "kernel": add_kernel, "occupancy": add_occupancy,
       "speed": add_speed,
       "scaling": add_scaling}


def measurement_plan(settings):
    the_plan = plan.Plan()
    for part in settings["parts"]:
        ADD[part](the_plan, settings)
    return the_plan


# ============================================================ reading the points back

def without_repeat(result):
    return plan.key(dict({name: result[name] for name in ("processor", "precision", "sms", "photons", "variant",
                                                          "star")}, repeat=0))


def grouped(results):
    groups = defaultdict(list)
    for result in results:
        groups[without_repeat(result)].append(result)
    return groups


def kernel_time(result):
    return sum(kernel["KERNEL_TIME"] for kernel in result["KERNEL"]) or result["SUMMARY"]["PHOTONS_TIME"]


def kernel_cycles(result):
    return sum(sms["CYCLES"] for sms in result["SMS"]) or None


def spread(values):
    middle = statistics.median(values)
    return {"median": middle, "min": min(values), "max": max(values),
            "spread": (max(values) - min(values)) / middle if middle else 0.0, "repeats": len(values)}


def monitor(results, name):
    values = [result["GPU_MONITOR"].get(name) for result in results if result["GPU_MONITOR"].get(name) is not None]
    return statistics.median(values) if values else None


def one_of(results, settings, precision, variant=None, sms=None):
    photons = reference(settings, "gpu", precision)
    for group in grouped(results).values():
        each = group[0]
        if (each["processor"], each["precision"], each["variant"], each["sms"], each["photons"]) == \
                ("gpu", precision, variant, sms, photons):
            return group
    raise LookupError(f"no gpu{precision} {variant or 'baseline'} run on {sms or 'all'} SMs at N = {photons:.1e}")


def run_row(group, baseline_group=None):
    first = group[0]["SUMMARY"]
    times = spread([kernel_time(result) for result in group])
    cycles = [kernel_cycles(result) for result in group if kernel_cycles(result)]
    row = {"photons": group[0]["photons"], "registers": first.get("REGISTERS_PER_THREAD"),
           "occupancy": first.get("OCCUPANCY"), "threads_per_block": first.get("THREADS_PER_BLOCK"),
           "blocks_per_sm": first.get("BLOCKS_PER_UNIT"), "kernel_time": times,
           "photons_per_second": group[0]["photons"] / times["median"],
           "cycles": statistics.median(cycles) if cycles else None,
           "effective_mhz": statistics.median(c / kernel_time(r) / 1e6 for r, c in zip(group, cycles)) if cycles else None,
           "sm_clock_mhz": monitor(group, "SM_CLOCK_MHZ_MEDIAN"), "power_w": monitor(group, "POWER_W_MEDIAN"),
           "power_cap_share": monitor(group, "POWER_CAP_SHARE"), "image_digest": group[0]["IMAGE_DIGEST"],
           "same_image_every_repeat": len({result["IMAGE_DIGEST"] for result in group}) == 1}
    if baseline_group:
        base = run_row(baseline_group)
        row["time_vs_baseline"] = times["median"] / base["kernel_time"]["median"]
        row["cycles_vs_baseline"] = row["cycles"] / base["cycles"] if row["cycles"] and base["cycles"] else None
        row["image_equals_baseline"] = row["image_digest"] == base["image_digest"]
    return row


# ============================================================ the analyses

def analyse_validation(folder, settings, results):
    found = []
    for name, processor, precision, photons in validation_checks(settings):
        config = engine.default_star_config(photons, **star(settings))
        result = checks.run_check(name, config, processor, precision,
                                  os.path.join(folder, "validation", f"{name}-{processor}{precision}"))
        found.append({key: result[key] for key in ("name", "processor", "precision", "passed", "headline", "data")})
        print(f"  validation {name}-{processor}{precision}: {result['headline']}", flush=True)
    digests = {f"{result['processor']}{result['precision']}": result["IMAGE_DIGEST"] for result in results
               if "digests" in result["parts"]}
    return {"checks": found, "digests": digests, "digest_photons": DIGEST_PHOTONS}


def analyse_kernel(folder, settings, results):
    data = {"resources": {}, "variants": {}, "sections": {}}
    for precision in engine.PRECISIONS:
        base = one_of(results, settings, precision)
        summary = base[0]["SUMMARY"]
        data["resources"][f"fp{precision}"] = {
            "registers": summary["REGISTERS_PER_THREAD"], "spill_bytes": device.spill_bytes(precision, None),
            "local_bytes": summary["LOCAL_BYTES_PER_THREAD"], "constant_bytes": summary["CONSTANT_BYTES"],
            "threads_per_block": summary["THREADS_PER_BLOCK"], "blocks_per_sm": summary["BLOCKS_PER_UNIT"],
            "occupancy": summary["OCCUPANCY"]}
        rows = [dict(run_row(base, base), variant="baseline", spill_bytes=device.spill_bytes(precision, None))]
        for variant in KERNEL_VARIANTS:
            group = one_of(results, settings, precision, variant)
            rows.append(dict(run_row(group, base), variant=variant,
                             spill_bytes=device.spill_bytes(precision, variant)))
        data["variants"][f"fp{precision}"] = rows
        sections = one_of(results, settings, precision, "sections")[0]["SECTIONS"]
        data["sections"][f"fp{precision}"] = [{"name": s["NAME"], "share": s["SHARE"]} for s in sections]
    data["instruction_mix"] = {name: mix["units"] for name, mix in
                               instruction_mix.mixes_as_data(instruction_mix.mixes_of_the_kernel()).items()}
    return data


def occupancy_figure(rows_by_precision, limits, path):
    fig, axes = plt.subplots(2, 1, figsize=(8, 7.5), sharex=True)
    registers = list(range(24, 129))
    axes[0].step(registers, [device.occupancy_allowed_by_registers(r, limits) for r in registers], where="post",
                 color=scaling.IDEAL, linewidth=1.2, label="what the registers allow")
    for precision in engine.PRECISIONS:
        rows = rows_by_precision[f"fp{precision}"]
        x = [row["registers"] for row in rows]
        axes[0].plot(x, [row["occupancy"] for row in rows], "o", color=COLORS[precision], markersize=8,
                     label=f"FP{precision}: the launch")
        axes[1].plot(x, [1 / row["cycles_vs_baseline"] for row in rows], "o-", color=COLORS[precision], linewidth=2,
                     markersize=7, label=f"FP{precision}")
    axes[1].axhline(1, color=scaling.IDEAL, linewidth=1, linestyle=":")
    scaling.styled_axes(axes[0], "Occupancy: the registers per thread decide how many warps fit", "",
                        f"occupancy (warps resident / {device.max_warps(limits)})")
    scaling.styled_axes(axes[1], "Speed in cycles (the clock removed), vs the compiler's own choice",
                        "registers per thread (after the cap)", "speed vs uncapped")
    for ax in axes:
        ax.legend(frameon=False, fontsize=8, labelcolor=scaling.TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def add_instruction_counts(rows, precision):
    for row in rows:
        cap = None if row["cap"] == "none" else row["cap"]
        row.update(instruction_mix.instruction_count(engine.binary_path("gpu", precision, register_variant(cap))))
    uncapped = rows[0]["instructions"]
    for row in rows:
        row["instructions_vs_uncapped"] = row["instructions"] / uncapped


def analyse_occupancy(folder, settings, results):
    limits = device.limits()
    rows = {}
    for precision, caps in REGISTER_CAPS.items():
        base = one_of(results, settings, precision)
        rows[f"fp{precision}"] = [
            dict(run_row(one_of(results, settings, precision, register_variant(cap)), base), cap=cap or "none",
                 spill_bytes=device.spill_bytes(precision, register_variant(cap)) or 0)
            for cap in caps]
        for row in rows[f"fp{precision}"]:
            row["warps_allowed_by_registers"] = device.warps_allowed_by_registers(row["registers"], limits)
        add_instruction_counts(rows[f"fp{precision}"], precision)
    occupancy_figure(rows, limits, os.path.join(folder, "figures", "registers_and_occupancy.png"))
    targets = [device.target_occupancy(target) for target in settings["targets"]]
    here = {f"fp{p}": device.occupancy_row(rows[f"fp{p}"][0]["registers"], rows[f"fp{p}"][0]["spill_bytes"], limits)
            for p in engine.PRECISIONS}
    return {"device": limits, "this_gpu": here, "rows": rows, "targets": targets,
            "figure": "figures/registers_and_occupancy.png"}


def speed_label(row):
    return f"FP{row['precision']}, {row['sms']} SM" + ("s" if row["sms"] > 1 else "")


def speed_figure(rows, path):
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    values = [row["photons_per_second"]["median"] for row in rows]
    ax.barh([speed_label(row) for row in rows], values, color=[COLORS[row["precision"]] for row in rows], height=0.5)
    ax.set_xscale("log")
    for row, value in enumerate(values):
        ax.text(value * 1.15, row, f"{value:.2g}", va="center", color=scaling.TEXT, fontsize=9)
    ax.invert_yaxis()
    scaling.styled_axes(ax, "Photons per second, photon phase: one SM and all SMs", "photons / s", "")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def speed_row(group, precision):
    cycles = [kernel_cycles(result) for result in group if kernel_cycles(result)]
    photon_times = [result["SUMMARY"]["PHOTONS_TIME"] for result in group]
    return {"precision": precision, "sms": group[0]["SUMMARY"]["COMPUTE_UNITS"], "photons": group[0]["photons"],
            "photons_per_second": spread([result["SUMMARY"]["PHOTONS_PER_SECOND"] for result in group]),
            "cycles": statistics.median(cycles) if cycles else None,
            "clock_during_photons_mhz": statistics.median(cycles) / statistics.median(photon_times) / 1e6
            if cycles else None,
            "sm_clock_mhz": monitor(group, "SM_CLOCK_MHZ_MEDIAN"), "power_w": monitor(group, "POWER_W_MEDIAN"),
            "power_cap_share": monitor(group, "POWER_CAP_SHARE")}


def analyse_speed(folder, settings, results):
    rows = [speed_row(one_of(results, settings, precision, sms=sms), precision)
            for precision in engine.PRECISIONS for sms in (1, None)]
    speed = {(row["precision"], row["sms"] == 1): row["photons_per_second"]["median"] for row in rows}
    every = all_sms()
    speed_figure(rows, os.path.join(folder, "figures", "speed.png"))
    return {"rows": rows, "all_sms": every,
            "all_sms_over_one_sm": {f"fp{p}": speed[(p, False)] / speed[(p, True)] for p in engine.PRECISIONS},
            "fp32_over_fp64": {"one_sm": speed[(32, True)] / speed[(64, True)],
                               "all_sms": speed[(32, False)] / speed[(64, False)]}}


def scaling_points(results, label):
    total = all_sms()
    return [dict(result, sms=result["sms"] or total) for result in results if label in result["parts"]]


def analyse_scaling(folder, settings, results):
    figures = os.path.join(folder, "figures")
    found = {}
    for precision in engine.PRECISIONS:
        found[f"strong-fp{precision}"] = scaling.analyse_strong(scaling_points(results, f"strong-fp{precision}"),
                                                                precision, figures, f"strong_fp{precision}")
        found[f"weak-fp{precision}"] = scaling.analyse_weak(scaling_points(results, f"weak-fp{precision}"),
                                                            precision, figures, f"weak_fp{precision}")
        scaling.plot_strong_vs_weak(found[f"strong-fp{precision}"], found[f"weak-fp{precision}"], precision, figures)
        scaling.plot_amdahl_ceiling(found[f"strong-fp{precision}"], precision, figures)
    found["problem-size-fp32"] = scaling.analyse_problem_size(scaling_points(results, "problem-size-fp32"), figures)
    return found


def measured_speed(results, settings, precision):
    try:
        group = one_of(results, settings, precision)
        return statistics.median(result["SUMMARY"]["PHOTONS_PER_SECOND"] for result in group), "measured in this run"
    except LookupError:
        return speeds.speed_of("gpu", precision)[0], "this host's stored speed"


def memory_for_image(results):
    gpu = [result for result in results if result["processor"] == "gpu"]
    if gpu:
        return gpu[0]["SUMMARY"]["MEMORY_FOR_IMAGE"]
    return plan.described("gpu", 32, None, None, "{}")["MEMORY_FOR_IMAGE"]


def compute_figure(rows, path):
    fig, (ax_photons, ax_blur) = plt.subplots(1, 2, figsize=(12, 4.6))
    for size, color in zip(COST_SIZES, (scaling.CYCLES, scaling.PHOTON_PHASE, scaling.WHOLE_RUN)):
        mine = [row for row in rows if row["size"] == size and row["observers"] == COST_OBSERVERS[0]]
        frames = [row["frames"] for row in mine]
        ax_photons.plot(frames, [row["photons"] for row in mine], "o-", color=color, linewidth=2, label=f"{size}²")
        ax_blur.plot(frames, [row["rotation_blur_px"] for row in mine], "o-", color=color, linewidth=2,
                     label=f"{size}²: rotation and equator")
        ax_blur.plot(frames, [row["pole_blur_px"] for row in mine], "o--", color=color, linewidth=1.2,
                     label=f"{size}²: polar bin")
    ax_photons.set_yscale("log")
    ax_blur.set_yscale("log")
    scaling.styled_axes(ax_photons, f"Photons for SNR {COST_SNR} at the limb: N ∝ nμ · frames · size²",
                        "frames (nμ = frames/π)", "photons N")
    scaling.styled_axes(ax_blur, "The blur those bins leave in the image", "frames (nμ = frames/π)", "blur (pixels)")
    ax_photons.legend(frameon=False, fontsize=8, labelcolor=scaling.TEXT)
    ax_blur.legend(frameon=False, fontsize=7, labelcolor=scaling.TEXT, loc="center right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def memory_figure(rows, memory, path):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    frames = 300
    width = 0.38
    for offset, observers, color in ((-width / 2, COST_OBSERVERS[0], scaling.PHOTON_PHASE),
                                     (width / 2, COST_OBSERVERS[1], scaling.WHOLE_RUN)):
        mine = [row for row in rows if row["frames"] == frames and row["observers"] == observers]
        x = [index + offset for index in range(len(mine))]
        bars = ax.bar(x, [row["bytes"] / 1e9 for row in mine], width * 0.9, color=color,
                      label=f"{observers} observers")
        for bar, row in zip(bars, mine):
            note = f"{row['passes']} pass" + ("es" if row["passes"] and row["passes"] > 1 else "") if row["passes"] \
                else "does not fit"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.08, note, ha="center", fontsize=8,
                    color=scaling.TEXT)
    ax.axhline(memory / 1e9, color=scaling.IDEAL, linestyle=":", linewidth=1.2,
               label=f"memory for the image on this GPU ({memory / 1e9:.1f} GB)")
    ax.set_xticks(range(len(COST_SIZES)), [f"{size}²" for size in COST_SIZES])
    ax.set_yscale("log")
    plain = matplotlib.ticker.FuncFormatter(lambda value, _: f"{value:g}")
    ax.yaxis.set_major_formatter(plain)
    ax.yaxis.set_minor_formatter(plain)
    scaling.styled_axes(ax, f"Memory of one run's image ({frames} frames): observers × frames × size² × 4 bytes",
                        "image size", "GB")
    ax.legend(frameon=False, fontsize=8, labelcolor=scaling.TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


LARGE_N = (10 ** 11, 10 ** 12, 10 ** 13)


def median_of(group, key):
    return statistics.median(result["SUMMARY"][key] for result in group)


def whole_run_model(results, settings, precision):
    one, every = one_of(results, settings, precision, sms=1), one_of(results, settings, precision)
    serial = statistics.median(result["SUMMARY"]["TIME"] - result["SUMMARY"]["PHOTONS_TIME"] for result in every)
    return {"one_sm": median_of(one, "PHOTONS_PER_SECOND"), "all_sms": median_of(every, "PHOTONS_PER_SECOND"),
            "serial_seconds": serial}


def predicted_scaling(model, photons):
    one = photons / model["one_sm"] + model["serial_seconds"]
    every = photons / model["all_sms"] + model["serial_seconds"]
    sweep = sum(photons / (model["one_sm"] * min(n, all_sms())) + model["serial_seconds"] for n in sm_counts())
    return {"photons": photons, "one_sm_seconds": one, "all_sms_seconds": every, "whole_run_speedup": one / every,
            "strong_sweep_seconds": sweep, "weak_sweep_seconds": len(sm_counts()) * every}


def measured_whole_run(results, precision):
    pairs = defaultdict(dict)
    for result in results:
        if result["processor"] == "gpu" and result["precision"] == precision and result["variant"] is None \
                and result["sms"] in (None, 1):
            pairs[result["photons"]].setdefault(result["sms"], []).append(result["SUMMARY"]["TIME"])
    return {photons: statistics.median(times[1]) / statistics.median(times[None])
            for photons, times in sorted(pairs.items()) if 1 in times and None in times}


def large_n_figure(models, measured, path):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    curve = [10 ** (9 + step / 20) for step in range(0, 101)]
    for precision, model in models.items():
        ax.plot(curve, [predicted_scaling(model, n)["whole_run_speedup"] for n in curve], color=COLORS[precision],
                linewidth=2, label=f"FP{precision}: predicted from the measured speeds and serial part")
        points = measured[precision]
        ax.plot(list(map(float, points)), list(points.values()), "o", color=COLORS[precision], markersize=8,
                label=f"FP{precision}: measured")
    ax.axhline(all_sms(), color=scaling.IDEAL, linestyle=":", linewidth=1.2, label=f"ideal: {all_sms()}×")
    for photons in LARGE_N:
        hours = predicted_scaling(models[32], photons)["one_sm_seconds"] / 3600
        ax.annotate(f"FP32 on 1 SM:\n{hours * 60:.0f} min" if hours < 1.5 else f"FP32 on 1 SM:\n{hours:.1f} h",
                    (photons, 3), ha="center", fontsize=8, color=scaling.TEXT)
    ax.set_xscale("log")
    ax.set_ylim(0, all_sms() * 1.1)
    scaling.styled_axes(ax, "The whole run (photons + saving the image) on all SMs vs one: the larger N, the closer "
                            "to ideal", "photons N", "speedup, whole run")
    ax.legend(frameon=False, fontsize=8, labelcolor=scaling.TEXT, loc="center right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def large_n(folder, settings, results):
    models = {precision: whole_run_model(results, settings, precision) for precision in engine.PRECISIONS}
    measured = {precision: measured_whole_run(results, precision) for precision in engine.PRECISIONS}
    large_n_figure(models, measured, os.path.join(folder, "figures", "costs_large_n.png"))
    rows = [dict(predicted_scaling(models[precision], photons), precision=precision)
            for precision in engine.PRECISIONS
            for photons in (reference(settings, "gpu", precision), *LARGE_N)]
    return {"models": {f"fp{p}": model for p, model in models.items()},
            "measured_whole_run": {f"fp{p}": {str(n): s for n, s in points.items()} for p, points in measured.items()},
            "rows": rows}


def analyse_costs(folder, settings, results):
    speed32, source = measured_speed(results, settings, 32)
    speed64, _ = measured_speed(results, settings, 64)
    memory = memory_for_image(results)
    rows = []
    for size in COST_SIZES:
        for frames in COST_FRAMES:
            for observers in COST_OBSERVERS:
                row = budget.configuration(size, frames, observers, COST_SNR, speed32, memory)
                row["seconds_fp64"] = row["passes"] * row["photons"] / speed64 if row["passes"] else None
                rows.append(row)
    compute_figure(rows, os.path.join(folder, "figures", "costs_compute.png"))
    memory_figure(rows, memory, os.path.join(folder, "figures", "costs_memory.png"))
    found = {"rows": rows, "snr": COST_SNR, "speed_fp32": speed32, "speed_fp64": speed64, "speed_source": source,
             "memory_for_image": memory}
    try:
        found["large_n"] = large_n(folder, settings, results)
    except LookupError:
        found["large_n"] = None
    return found


ANALYSE = {"costs": analyse_costs, "validation": analyse_validation, "kernel": analyse_kernel, "occupancy": analyse_occupancy,
           "speed": analyse_speed, "scaling": analyse_scaling}
