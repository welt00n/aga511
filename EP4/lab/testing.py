import os
import socket
import time

from lab import checks, engine, environment, gpu_monitor, plan, runs, speeds

PHOTONS = 10 ** 7
CPU_ONLY = ("build", "rng cpu", "run cpu64", "run cpu32", "split cpu64")
RUNS_PER_TEST = {"build": 0, "split cpu64": 3, "split gpu64": 3, "passes gpu64": 2, "gpu = cpu": 2, "gpu ~ cpu": 2}
BUILD_SECONDS = 30


def build(folder):
    binaries = engine.build_all()
    return True, f"{len(binaries)} binaries built", {"binaries": [os.path.basename(path) for path in binaries]}


def random_numbers(processor):
    def test(folder):
        engine.engine_output([engine.build_engine(processor, 64), "--test-rng"])
        return True, "the numbers match the reference" if processor == "cpu" else "cuRAND = the CPU's copy", {}
    return test


def one_run(processor, precision):
    def test(folder):
        os.makedirs(os.path.dirname(folder), exist_ok=True)
        summary = checks.run(engine.default_star_config(PHOTONS), folder, processor, precision)
        checks.delete(folder)
        passed = summary["PHOTONS"] == PHOTONS and summary["RECORDED"] > 0
        return passed, f"{summary['RECORDED']} of {PHOTONS:.0e} photons recorded", {"summary": summary}
    return test


def check(name, processor, precision):
    def test(folder):
        result = checks.run_check(name, engine.default_star_config(PHOTONS), processor, precision, folder)
        return result["passed"] is not False, result["headline"], result["data"]
    return test


TESTS = [
    ("build", "every engine compiles here", build),
    ("rng cpu", "Philox on the CPU gives the published numbers", random_numbers("cpu")),
    ("rng gpu", "cuRAND's Philox on the GPU = the CPU's, bit for bit", random_numbers("gpu")),
    ("run cpu64", "the CPU engine runs (FP64)", one_run("cpu", 64)),
    ("run cpu32", "the CPU engine runs (FP32)", one_run("cpu", 32)),
    ("run gpu64", "the GPU engine runs (FP64)", one_run("gpu", 64)),
    ("run gpu32", "the GPU engine runs (FP32)", one_run("gpu", 32)),
    ("split cpu64", "any split of the chunks gives the same image (CPU)", check("split", "cpu", 64)),
    ("split gpu64", "any split of the chunks gives the same image (GPU)", check("split", "gpu", 64)),
    ("passes gpu64", "any number of passes gives the same image (GPU)", check("passes", "gpu", 64)),
    ("gpu = cpu", "the GPU's image = the CPU's, bit for bit (FP64)", check("gpu", "gpu", 64)),
    ("gpu ~ cpu", "the GPU and the sequential CPU agree with independent photons (χ²)",
     check("independent", "gpu", 64)),
]


def chosen_tests():
    if gpu_monitor.gpu_is_visible():
        return TESTS
    return [test for test in TESTS if test[0] in CPU_ONLY]


def predicted_seconds(tests):
    cpu_seconds = PHOTONS / speeds.speed_of("cpu", 64)[0]
    return BUILD_SECONDS + sum(RUNS_PER_TEST.get(name, 1) * (cpu_seconds if "cpu" in name else 1.0)
                               for name, _, _ in tests)


def run_one(name, proves, test, folder):
    started_at = time.perf_counter()
    try:
        passed, detail, data = test(os.path.join(folder, name.replace(" ", "-").replace("=", "is").replace("~", "like")))
    except Exception as error:
        passed, detail, data = False, f"error: {error}".splitlines()[0], {}
    seconds = time.perf_counter() - started_at
    print(f"  [{'pass' if passed else 'FAIL'}] {name:12s} {proves}: {detail} ({seconds:.1f} s)", flush=True)
    return {"name": name, "proves": proves, "passed": bool(passed), "detail": detail, "seconds": seconds,
            "data": data}


def result_lines(results):
    lines = ["| test | proves | result | detail | s |", "|---|---|---|---|---|"]
    for result in results:
        lines.append(f"| {result['name']} | {result['proves']} | {'pass' if result['passed'] else '**FAIL**'} | "
                     f"{result['detail']} | {result['seconds']:.1f} |")
    return lines + ["", "Correctness only: these runs are far below the minimum N, so their times are not "
                        "measurements. `measure` makes those."]


def test(yes=False, plan_only=False):
    found = environment.checks_for(needs_gpu=gpu_monitor.gpu_is_visible())
    if not environment.show(found):
        return False
    environment.show_launch()
    tests = chosen_tests()
    print(f"\nThe tests ({len(tests)}, N = {PHOTONS:.0e}, ~{plan.duration(predicted_seconds(tests))}):")
    for name, proves, _ in tests:
        print(f"  {name:12s} {proves}")
    if plan_only or not plan.confirmed(yes):
        return True
    with runs.Run("test", socket.gethostname(), {"photons": PHOTONS, "tests": [t[0] for t in tests]}) as run:
        print()
        results = [run_one(name, proves, test, run.path("checks")) for name, proves, test in tests]
        runs.save_json({"environment": found, "results": results}, run.path("test.json"))
        for result in results:
            if "summary" in result["data"]:
                run.add_machine(result["data"]["summary"])
        failed = [result["name"] for result in results if not result["passed"]]
        headline = (f"all {len(results)} tests passed" if not failed else
                    f"{len(failed)} of {len(results)} tests FAILED: {', '.join(failed)}")
        run.finish(headline, result_lines(results), passed=not failed)
    return not failed
