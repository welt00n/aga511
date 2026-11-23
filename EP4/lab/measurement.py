import os
import socket

from lab import device, engine, environment, explainer, gpu_monitor, parts, plan, points, report, runs

ON_ONE_AND_ALL_SMS = [("gpu", precision, sms) for precision in engine.PRECISIONS for sms in (None, 1)]


def new_settings(chosen, seconds, repeats, targets, allow_invalid, size, frames):
    return {"parts": list(chosen), "seconds": seconds, "repeats": repeats, "targets": list(targets),
            "allow_invalid": allow_invalid, "photons": parts.reference_photons(seconds),
            "star": {"size": size, "frames": frames}}


def minimum_row(processor, precision, sms):
    return dict(plan.minimum_photons(processor, precision, sms), binary=f"{processor}{precision}",
                sms=sms or parts.all_sms())


def machine_record(found):
    driver = gpu_monitor.query_now(["driver_version"])
    return {"gpu": device.limits(), "nvcc": engine.nvcc_version().strip().splitlines()[-1],
            "driver": driver[0] if driver else None, "environment": found,
            "minimums": [minimum_row(*each) for each in ON_ONE_AND_ALL_SMS]}


def add_cpu_name(machine, results):
    cpu = [result for result in results if result["processor"] == "cpu"]
    if cpu:
        machine["cpu"] = cpu[0]["SUMMARY"]["PROCESSOR_NAME"].replace("_", " ")


def analysed(folder, settings, results):
    found = {}
    for name in settings["parts"]:
        path = os.path.join(folder, "parts", name + ".json")
        if name == "validation" and os.path.exists(path):
            found[name] = runs.read_json(path)
            continue
        print(f"analysing {name}", flush=True)
        found[name] = parts.ANALYSE[name](folder, settings, results)
        runs.save_json(found[name], path)
    return found


def headline(found, valid):
    pieces = []
    if "speed" in found:
        speed = found["speed"]
        every = {row["precision"]: row["photons_per_second"]["median"] for row in speed["rows"] if row["sms"] > 1}
        pieces.append(f"GPU {every[64]:.2g}/s FP64 ({speed['all_sms_over_one_sm']['fp64']:.1f}× one SM), "
                      f"{every[32]:.2g}/s FP32 ({speed['all_sms_over_one_sm']['fp32']:.1f}× one SM)")
    if "scaling" in found:
        strong = found["scaling"]["strong-fp64"]
        every = max(strong["sms"])
        pieces.append(f"strong FP64 {strong['speedup_photon_phase'][every]:.1f}× on {every} SMs")
    if "occupancy" in found:
        rows = found["occupancy"]["rows"]["fp64"]
        best = min(rows, key=lambda row: row["cycles_vs_baseline"] or 9)
        pieces.append(f"FP64 fastest at {best['registers']} registers (occupancy {best['occupancy']:.2f})")
    if "validation" in found:
        passed = [check["passed"] for check in found["validation"]["checks"] if check["passed"] is not None]
        pieces.append("validation passed" if all(passed) else "validation FAILED")
    text = "; ".join(pieces) or "measured"
    return text if valid else "NOT VALID (below the minimum N): " + text


def summary_lines(found, document):
    lines = [f"**The report:** [{os.path.basename(document)}]({os.path.basename(document)}) and "
             "[report.ipynb](report.ipynb) (the same, with the code that reads the data).", ""]
    for name in found:
        lines.append(f"- `{name}`: [`parts/{name}.json`](parts/{name}.json)")
    return lines


def figures_of(folder):
    names = sorted(name for name in os.listdir(os.path.join(folder, "figures")) if name.endswith(".png"))
    return [f"figures/{name}" for name in names]


def label_of(settings):
    chosen = "all" if set(settings["parts"]) == set(parts.PARTS) else "-".join(settings["parts"])
    return f"{socket.gethostname()}-{chosen}"


def run_measurement(run, settings, the_plan, found_environment, valid):
    os.makedirs(run.path("figures"), exist_ok=True)
    os.makedirs(run.path("parts"), exist_ok=True)
    machine = machine_record(found_environment)
    the_plan.save(run.path("plan.json"))
    points.run_missing(the_plan, run.folder)
    results = [dict(points.read(run.folder, name), parts=entry["parts"]) for name, entry in the_plan.entries.items()]
    for result in results[:1] + [result for result in results if result["processor"] == "cpu"][:1]:
        run.add_machine(result["SUMMARY"])
    add_cpu_name(machine, results)
    runs.save_json(machine, run.path("machine.json"))
    found = analysed(run.folder, settings, results)
    print("writing the report", flush=True)
    cells = explainer.measurement_cells(run.folder, run.record, machine, valid)
    _, document = report.write_report(f"EP4 measurement: {machine['gpu']['NAME'] if machine['gpu'] else run.record['host']}",
                                      cells, run.folder)
    run.finish(headline(found, valid), summary_lines(found, document), figures_of(run.folder), passed=valid or None)


def measure(chosen, seconds=5.0, repeats=3, targets=("80",), yes=False, plan_only=False, allow_invalid=False,
            continue_folder=None, size=1024, frames=300):
    found_environment = environment.checks_for(needs_report=True)
    if not environment.show(found_environment):
        return False
    environment.show_launch()
    record = runs.read_json(os.path.join(continue_folder, "run.json")) if continue_folder else None
    settings = record["asked"] if record else new_settings(chosen, seconds, repeats, targets, allow_invalid, size,
                                                           frames)
    the_plan = parts.measurement_plan(settings)
    plan.show("The measurement", the_plan, ON_ONE_AND_ALL_SMS)
    valid = not the_plan.invalid()
    if not valid and not settings["allow_invalid"]:
        print("\nNot run: a measurement with runs below the minimum N would give no conclusions. Use `test` to try "
              "things out, raise --seconds, or pass --allow-invalid to make an explicitly invalid report.")
        return False
    if plan_only or not plan.confirmed(yes):
        return True
    with (runs.Run.reopen(continue_folder) if continue_folder else
          runs.Run("measure", label_of(settings), settings)) as run:
        run_measurement(run, settings, the_plan, found_environment, valid)
    return True
