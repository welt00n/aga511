import hashlib
import json
import os
import sys

import common
from lab import engine

PHOTONS = 10 ** 7
REFERENCE = os.path.join(common.RESULTS_FOLDER, "bit_for_bit_reference.json")


def image_digest(image):
    return hashlib.sha256(image.tobytes()).hexdigest()[:16]


def fingerprint(processor, precision):
    folder = common.scratch_folder("reference")
    summary = engine.lines_of_kind(engine.engine_output(engine.engine_command(
        engine.default_star_config(PHOTONS), folder, processor, precision, None)), "SUMMARY")[0]
    image = engine.load_and_delete_observers(folder)[1]
    resources = (list(common.kernel_resources(engine.read_text(engine.binary_path(processor, precision) + ".report")))
                 if processor == "gpu" else None)
    return {"image": image_digest(image), "recorded": summary["RECORDED"], "blocked": summary["BLOCKED"],
            "registers_and_spills": resources}


def fingerprints():
    return {f"{processor}{precision}": fingerprint(processor, precision)
            for processor in engine.PROCESSORS for precision in engine.PRECISIONS}


def compare(saved, now):
    same = True
    for name, values in now.items():
        verdict = "same" if saved.get(name) == values else f"DIFFERENT (saved {saved.get(name)})"
        same &= verdict == "same"
        print(f"  {name}: {values} -> {verdict}")
    return same


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "compare"
    print(f"A fixed run (N = {PHOTONS:.0e}, the default star) on every binary: the image's digest, the counts, and")
    print("the GPU kernel's registers. 'save' before a refactor, 'compare' after: the same means nothing changed.\n")
    now = fingerprints()
    if mode == "save" or not os.path.exists(REFERENCE):
        os.makedirs(common.RESULTS_FOLDER, exist_ok=True)
        engine.write_text(REFERENCE, json.dumps(now, indent=1))
        for name, values in now.items():
            print(f"  {name}: {values}")
        print(f"\nsaved as the reference in {os.path.relpath(REFERENCE, common.EP4_FOLDER)}")
    else:
        saved = json.loads(engine.read_text(REFERENCE))
        sys.exit(0 if compare(saved, now) else 1)
