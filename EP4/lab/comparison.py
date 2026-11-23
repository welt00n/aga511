import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from lab import explainer, report, runs, scaling  # noqa: E402
from lab.report import code, markdown  # noqa: E402

MACHINE_COLORS = [scaling.PHOTON_PHASE, scaling.WHOLE_RUN, scaling.CYCLES, "#8b5cf6"]


# ============================================================ reading the measurements

def measurement(folder):
    record = runs.read_json(os.path.join(folder, "run.json"))
    if record["kind"] != "measure" or record["status"] != "done":
        raise ValueError(f"{folder} is not a finished measurement ({record['kind']}, {record['status']})")
    machine = runs.read_json(os.path.join(folder, "machine.json"))
    parts = explainer.load_parts(folder, explainer.ORDER)
    name = (machine["gpu"] or {}).get("NAME", record["host"])
    return {"folder": os.path.relpath(folder, runs.RUNS_FOLDER), "name": name, "record": record, "machine": machine,
            "parts": parts}


def baseline_row(each, precision):
    rows = each["parts"].get("kernel", {}).get("variants", {}).get(f"fp{precision}", [])
    return next((row for row in rows if row["variant"] == "baseline"), None)


def speed_of(each, precision, one_sm=False):
    rows = each["parts"].get("speed", {}).get("rows", [])
    return next((row["photons_per_second"]["median"] for row in rows
                 if row["precision"] == precision and (row["sms"] == 1) == one_sm), None)


def machine_row(each):
    gpu = each["machine"]["gpu"] or {}
    return {"name": each["name"], "folder": each["folder"], "host": each["record"]["host"],
            "code": each["record"]["code_hash"], "started": each["record"]["started"], "sms": gpu.get("SMS"),
            "cc": gpu.get("COMPUTE_CAPABILITY"), "threads_per_sm": gpu.get("MAX_THREADS_PER_SM"),
            "registers_per_sm": gpu.get("REGISTERS_PER_SM"),
            "driver": each["machine"].get("driver"), "nvcc": each["machine"].get("nvcc")}


def speed_rows(measured):
    rows = []
    for each in measured:
        sms = (each["machine"]["gpu"] or {}).get("SMS")
        for precision in (64, 32):
            speed, base = speed_of(each, precision), baseline_row(each, precision)
            clock = base["effective_mhz"] if base else None
            rows.append({"machine": each["name"], "precision": precision, "photons_per_second": speed,
                         "per_sm": speed / sms if speed and sms else None,
                         "per_sm_per_ghz": speed / sms / (clock / 1e3) if speed and sms and clock else None,
                         "clock_mhz": clock, "power_w": base["power_w"] if base else None,
                         "power_cap_share": base["power_cap_share"] if base else None,
                         "one_sm": speed_of(each, precision, one_sm=True)})
    return rows


def scaling_rows(measured):
    rows = []
    for each in measured:
        found = each["parts"].get("scaling")
        if not found:
            continue
        for precision in (64, 32):
            strong, weak = found[f"strong-fp{precision}"], found[f"weak-fp{precision}"]
            every = str(max(strong["sms"]))
            rows.append({"machine": each["name"], "precision": precision, "sms": int(every),
                         "strong_speedup": strong["speedup_photon_phase"][every],
                         "strong_p": strong["amdahl_photon_phase"]["p"], "whole_p": strong["amdahl_whole_run"]["p"],
                         "cycles_speedup": strong["speedup_cycles"].get(every),
                         "weak_efficiency": weak["efficiency_photon_phase"][every],
                         "s_prime": weak["gustafson_whole_run"]["s_prime"]})
    return rows


def occupancy_rows(measured):
    rows = []
    for each in measured:
        found = each["parts"].get("occupancy")
        if not found:
            continue
        for precision in ("fp64", "fp32"):
            caps = found["rows"][precision]
            best = min(caps, key=lambda row: row["cycles_vs_baseline"] or 9)
            rows.append({"machine": each["name"], "precision": precision, "registers": caps[0]["registers"],
                         "occupancy": caps[0]["occupancy"], "best_registers": best["registers"],
                         "best_occupancy": best["occupancy"], "best_cycles": best["cycles_vs_baseline"]})
    return rows


def digest_rows(measured):
    binaries = sorted({binary for each in measured for binary in each["parts"].get("validation", {})
                      .get("digests", {})})
    rows = []
    for binary in binaries:
        digests = [each["parts"].get("validation", {}).get("digests", {}).get(binary) for each in measured]
        rows.append({"binary": binary, **{each["name"]: digest for each, digest in zip(measured, digests)},
                     "identical": len(set(digests)) == 1 and None not in digests})
    return rows


# ============================================================ figures

def speed_figure(rows, names, path):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    width = 0.8 / len(names)
    for index, name in enumerate(names):
        values = [next((row["photons_per_second"] or 0 for row in rows
                        if row["machine"] == name and row["precision"] == precision), 0) for precision in (64, 32)]
        ax.bar(np.arange(2) + index * width, values, width * 0.9, color=MACHINE_COLORS[index % 4], label=name)
    ax.set_xticks(np.arange(2) + width * (len(names) - 1) / 2, ["FP64", "FP32"])
    ax.set_yscale("log")
    scaling.styled_axes(ax, "Photons per second on all SMs", "", "photons / s")
    ax.legend(frameon=False, fontsize=8, labelcolor=scaling.TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def strong_figure(measured, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, precision in zip(axes, (64, 32)):
        for index, each in enumerate(each for each in measured if "scaling" in each["parts"]):
            strong = each["parts"]["scaling"][f"strong-fp{precision}"]
            ax.plot(strong["sms"], [strong["speedup_photon_phase"][str(n)] for n in strong["sms"]], "o-",
                    color=MACHINE_COLORS[index % 4], linewidth=2, label=each["name"])
        top = max([max(each["parts"]["scaling"]["strong-fp64"]["sms"]) for each in measured
                   if "scaling" in each["parts"]] or [1])
        ax.plot([1, top], [1, top], color=scaling.IDEAL, linestyle=":", linewidth=1, label="ideal")
        scaling.styled_axes(ax, f"Strong scaling, FP{precision}, photon phase", "SMs", "speedup")
        ax.legend(frameon=False, fontsize=8, labelcolor=scaling.TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def occupancy_figure(measured, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, precision in zip(axes, ("fp64", "fp32")):
        for index, each in enumerate(each for each in measured if "occupancy" in each["parts"]):
            rows = each["parts"]["occupancy"]["rows"][precision]
            ax.plot([row["registers"] for row in rows], [1 / row["cycles_vs_baseline"] for row in rows], "o-",
                    color=MACHINE_COLORS[index % 4], linewidth=2, label=each["name"])
        ax.axhline(1, color=scaling.IDEAL, linestyle=":", linewidth=1)
        scaling.styled_axes(ax, f"{precision.upper()}: speed in cycles vs the compiler's choice",
                            "registers per thread (capped)", "speed vs uncapped")
        ax.legend(frameon=False, fontsize=8, labelcolor=scaling.TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ============================================================ the report

def cells(data):
    names = [machine["name"] for machine in data["machines"]]
    same_code = len({machine["code"] for machine in data["machines"]}) == 1
    speeds = {(row["machine"], row["precision"]): row for row in data["speeds"]}
    first = names[0]
    ratio_lines = "\n".join(
        f"- {name} vs {first}, FP{p}: {speeds[(name, p)]['photons_per_second'] / speeds[(first, p)]['photons_per_second']:.2f}× "
        f"the photons per second; per SM per GHz "
        + (f"{speeds[(name, p)]['per_sm_per_ghz'] / speeds[(first, p)]['per_sm_per_ghz']:.2f}×"
           if speeds[(name, p)]['per_sm_per_ghz'] and speeds[(first, p)]['per_sm_per_ghz'] else "—")
        for name in names[1:] for p in (64, 32)
        if speeds.get((name, p), {}).get("photons_per_second") and speeds.get((first, p), {}).get("photons_per_second"))
    digests = data["digests"]
    digest_line = ("- " + "; ".join(f"{row['binary']}: {'identical' if row['identical'] else 'different'}"
                                    for row in digests)) if digests else "- (no digests)"
    return [markdown(f"""
# EP4 comparison: {' vs '.join(names)}

**What this is.** The measurements of several machines side by side, read from their own folders (listed below).
Each one ran the same `measure` with the same engine code{'' if same_code else ' — **NOT the same engine code: see the table**'}.
"""), code("""
data = load("comparison")
table(data["machines"], [("machine", "name", None), ("SMs", "sms", None), ("cc", "cc", None),
                         ("threads/SM", "threads_per_sm", None), ("registers/SM", "registers_per_sm", None),
                         ("code", "code", None), ("driver", "driver", None), ("folder", "folder", None)])
"""), markdown("""
## Speed

Photons per second on all SMs (the photon phase); per SM; and per SM per GHz, which removes the number of SMs and the
clock and leaves what one SM does per cycle.
"""), code("""
table(data["speeds"], [("machine", "machine", None), ("FP", "precision", None),
                       ("photons/s", "photons_per_second", ".3g"), ("per SM", "per_sm", ".3g"),
                       ("per SM per GHz", "per_sm_per_ghz", ".3g"), ("clock (MHz)", "clock_mhz", ".0f"),
                       ("power (W)", "power_w", ".0f"), ("power cap", "power_cap_share", ".0%"),
                       ("one SM", "one_sm", ".3g")])
figure("figures/speeds.png")
"""), markdown("**What it means.**\n" + (ratio_lines or "- one machine only")), markdown("## Scaling"), code("""
table(data["scaling"], [("machine", "machine", None), ("FP", "precision", None), ("SMs", "sms", None),
                        ("strong speedup", "strong_speedup", ".1f"), ("Amdahl p (photons)", "strong_p", ".4f"),
                        ("Amdahl p (whole run)", "whole_p", ".4f"), ("speedup in cycles", "cycles_speedup", ".1f"),
                        ("weak efficiency", "weak_efficiency", ".3f"), ("Gustafson s′", "s_prime", ".4f")])
figure("figures/strong.png")
"""), markdown("## Registers and occupancy"), code("""
table(data["occupancy"], [("machine", "machine", None), ("precision", "precision", None),
                          ("registers", "registers", None), ("occupancy", "occupancy", ".2f"),
                          ("fewest cycles at", "best_registers", None), ("its occupancy", "best_occupancy", ".2f"),
                          ("its cycles vs uncapped", "best_cycles", ".3f")])
figure("figures/occupancy.png")
"""), markdown("## The same images?\n\nA fixed run (N = 10⁸, the measured star) on each machine, its images hashed. "
               "Identical hashes mean identical images, bit for bit.\n\n" + digest_line), code("""
table(data["digests"], [("binary", "binary", None)] + [(m["name"], m["name"], None) for m in data["machines"]]
      + [("identical", "identical", lambda v: "yes" if v else "NO")])
""")]


def unique_names(measured):
    plain = [each["name"] for each in measured]
    for each in measured:
        if plain.count(each["name"]) > 1:
            each["name"] = f"{each['name']} ({each['record']['started'].replace('T', ' ')[:16]})"


def compare(folders):
    measured = [measurement(folder) for folder in folders]
    unique_names(measured)
    names = [each["name"] for each in measured]
    data = {"machines": [machine_row(each) for each in measured], "speeds": speed_rows(measured),
            "scaling": scaling_rows(measured), "occupancy": occupancy_rows(measured), "digests": digest_rows(measured)}
    label = "-vs-".join(each["record"]["host"] for each in measured)
    with runs.Run("compare", label, {"measurements": [each["folder"] for each in measured]}) as run:
        os.makedirs(run.path("parts"), exist_ok=True)
        os.makedirs(run.path("figures"), exist_ok=True)
        runs.save_json(data, run.path("parts", "comparison.json"))
        speed_figure(data["speeds"], names, run.path("figures", "speeds.png"))
        strong_figure(measured, run.path("figures", "strong.png"))
        occupancy_figure(measured, run.path("figures", "occupancy.png"))
        _, document = report.write_report("EP4 comparison: " + " vs ".join(names), cells(data), run.folder)
        lines = [f"**The report:** [{os.path.basename(document)}]({os.path.basename(document)}), "
                 "[report.ipynb](report.ipynb)", ""] + [f"- {row['name']}: `{row['folder']}`" for row in data["machines"]]
        run.finish(f"{' vs '.join(names)}: see the report", lines,
                   ["figures/speeds.png", "figures/strong.png", "figures/occupancy.png"])
    return True
