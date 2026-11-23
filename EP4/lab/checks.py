import os
import shutil

import numpy as np

from lab import analysis, engine
from lab.star import describe_star, phases_table
from lab.videos import observer_frames

COUNTS = ("PHOTONS", "BLOCKED", "MISSED", "RECORDED")
FRAMES_AT_ONCE = 25


# ============================================================ running, and reading the images a few frames at a time

def run(config, folder, processor, precision, chunk_range=None):
    return engine.run_engine(config, folder, processor, precision, chunk_range)


def frame_batches(folder, record):
    frames = observer_frames(folder, record)
    for start in range(0, len(frames), FRAMES_AT_ONCE):
        yield np.asarray(frames[start:start + FRAMES_AT_ONCE], dtype=np.int64)


def observers_side_by_side(*folders):
    records = [engine.observer_records(folder) for folder in folders]
    return [list(zip(folders, same_observer)) for same_observer in zip(*records)]


def frames_summed(folder):
    return np.stack([sum(batch.sum(axis=0) for batch in frame_batches(folder, record))
                     for record in engine.observer_records(folder)])


def comparison(folder_a, folder_b):
    differing, moved, total = 0, 0, 0
    for (_, a_record), (_, b_record) in observers_side_by_side(folder_a, folder_b):
        for a, b in zip(frame_batches(folder_a, a_record), frame_batches(folder_b, b_record)):
            differing += int(np.count_nonzero(a != b))
            moved += int(np.abs(a - b).sum())
            total += int(a.sum())
    return {"identical": differing == 0, "differing_pixels": differing, "moved": moved / 2 / max(total, 1)}


def halves_add_up(first, second, full):
    for (_, a), (_, b), (_, c) in observers_side_by_side(first, second, full):
        for x, y, z in zip(frame_batches(first, a), frame_batches(second, b), frame_batches(full, c)):
            if not np.array_equal(x + y, z):
                return False
    return True


def delete(*folders):
    for folder in folders:
        shutil.rmtree(folder, ignore_errors=True)


def differing_counts(a, b):
    return [f"{key} {a[key]} vs {b[key]}" for key in COUNTS if a[key] != b[key]]


def outcome(passed, headline, results, figures=(), summaries=(), data=None):
    return {"passed": passed, "headline": headline, "results": results, "figures": list(figures),
            "summaries": list(summaries), "data": data or {}}


# ============================================================ the GPU vs the sequential CPU, independent photons

def compare_independent(config, processor, precision, folder):
    both = dict(config, photons=2 * config["photons"])
    chunks = engine.describe_engine(both, "cpu", precision)[0]["CHUNKS"]
    cpu_chunks, gpu_chunks = (0, chunks // 2), (chunks // 2, chunks)
    cpu_folder, gpu_folder = os.path.join(folder, "cpu"), os.path.join(folder, "gpu")
    cpu = run(both, cpu_folder, "cpu", precision, cpu_chunks)
    gpu = run(both, gpu_folder, "gpu", precision, gpu_chunks)
    gpu_image, cpu_image = frames_summed(gpu_folder), frames_summed(cpu_folder)
    delete(cpu_folder, gpu_folder)
    block = analysis.block_side(config["size"])
    chi2 = analysis.chi2_between_independent_images(analysis.in_blocks(gpu_image, block),
                                                    analysis.in_blocks(cpu_image, block))
    passed = abs(chi2["sigma_from_1"]) < 3
    analysis.plot_independent_comparison(gpu_image, cpu_image, chi2, (f"GPU, chunks {gpu_chunks}",
                                                                       f"CPU (sequential), chunks {cpu_chunks}"),
                                         "The GPU vs the sequential CPU, independent photons of the same star",
                                         os.path.join(folder, "comparison.png"))
    results = [f"The same star (seed {config['seed']}: the same spots), different chunks: chunks {cpu_chunks} on the "
               f"CPU, {gpu_chunks} on the GPU. Each chunk is its own random stream, so the photons are independent.",
               "", describe_star(config), "",
               *phases_table([(f"GPU, chunks {gpu_chunks}", gpu), (f"CPU (sequential), chunks {cpu_chunks}", cpu)]), "",
               f"- χ²/dof = {chi2['chi2_per_dof']:.4f} over {chi2['dof']} blocks of {block}×{block} pixels "
               f"({chi2['sigma_from_1']:+.2f} σ from 1)",
               "", "| observer | GPU | CPU | difference (σ) |", "|---|---|---|---|"]
    totals = []
    for observer, (a, b) in enumerate(zip(gpu_image.sum(axis=(1, 2)), cpu_image.sum(axis=(1, 2)))):
        sigma = float((int(a) - int(b)) / np.sqrt(a + b))
        results.append(f"| {observer} | {a} | {b} | {sigma:+.2f} |")
        totals.append({"observer": observer, "gpu": int(a), "cpu": int(b), "difference_sigma": sigma})
    headline = (f"χ²/dof = {chi2['chi2_per_dof']:.4f}: "
                + ("the same physics within noise" if passed else "THE IMAGES DIFFER"))
    data = {"photons": config["photons"], "precision": precision, "seed": config["seed"],
            "chunks": {"cpu": list(cpu_chunks), "gpu": list(gpu_chunks)},
            "chi2_per_dof": chi2["chi2_per_dof"], "dof": chi2["dof"], "sigma_from_1": chi2["sigma_from_1"],
            "block_pixels": block, "observer_totals": totals}
    return outcome(passed, headline, results, ["comparison.png"], [cpu, gpu], data)


# ============================================================ any split of the chunks, the same image

def check_split(config, processor, precision, folder):
    full_folder, first_folder, second_folder = (os.path.join(folder, name) for name in ("full", "first", "second"))
    full = run(config, full_folder, processor, precision)
    chunks = full["CHUNKS"]
    middle = chunks // 2
    first = run(config, first_folder, processor, precision, (0, middle))
    second = run(config, second_folder, processor, precision, (middle, chunks))
    identical = halves_add_up(first_folder, second_folder, full_folder)
    delete(full_folder, first_folder, second_folder)
    results = [describe_star(config) + f", chunk size {config['chunk_size']}", "",
               *phases_table([(f"[0, {chunks})", full), (f"[0, {middle})", first), (f"[{middle}, {chunks})", second)]),
               "", f"- recorded: {first['RECORDED']} + {second['RECORDED']} = "
                   f"{first['RECORDED'] + second['RECORDED']} vs {full['RECORDED']}"]
    headline = ("two halves = the full run, bit for bit" if identical else "the halves DIFFER from the full run")
    data = {"identical": bool(identical), "chunks": chunks,
            "recorded": {"first": first["RECORDED"], "second": second["RECORDED"], "full": full["RECORDED"]}}
    return outcome(identical, headline, results, summaries=[full], data=data)


# ============================================================ any number of passes, the same image

def check_passes(config, processor, precision, folder):
    one_folder, many_folder = os.path.join(folder, "one-pass"), os.path.join(folder, "passes")
    one_pass = run(dict(config, observers_per_pass=None), one_folder, processor, precision)
    passes = run(dict(config, observers_per_pass=1), many_folder, processor, precision)
    identical = comparison(one_folder, many_folder)["identical"]
    delete(one_folder, many_folder)
    same_counts = not differing_counts(one_pass, passes)
    results = [describe_star(config), "",
               *phases_table([("1 pass", one_pass), (f"{passes['PASSES']} passes of 1 observer", passes)]), "",
               "- counts: " + ("the same" if same_counts else ", ".join(differing_counts(one_pass, passes)))]
    headline = (f"1 pass = {passes['PASSES']} passes, bit for bit" if identical and same_counts
                else "the passes DIFFER from one pass")
    data = {"identical": bool(identical), "same_counts": same_counts, "passes": passes["PASSES"]}
    return outcome(identical and same_counts, headline, results, summaries=[one_pass], data=data)


# ============================================================ FP32 vs FP64, the same photons

def compare_precision(config, processor, precision, folder):
    fp64_folder, fp32_folder = os.path.join(folder, "fp64"), os.path.join(folder, "fp32")
    fp64 = run(config, fp64_folder, processor, 64)
    fp32 = run(config, fp32_folder, processor, 32)
    moved = comparison(fp64_folder, fp32_folder)["moved"]
    noise = analysis.relative_noise_per_lit_pixel(frames_summed(fp64_folder))
    delete(fp64_folder, fp32_folder)
    ratio = fp32["PHOTONS_PER_SECOND"] / fp64["PHOTONS_PER_SECOND"]
    results = ["The same seed: the same random numbers, so the same photons.", "", describe_star(config), "",
               *phases_table([("FP64", fp64), ("FP32", fp32)]), "",
               f"- FP32 / FP64 speed: {ratio:.2f}",
               f"- photons that land in another pixel in FP32: {moved:.2e}",
               f"- Poisson noise per lit pixel (frames summed): {noise:.2e}"]
    headline = f"FP32 moves {moved:.2e} of the photons (noise {noise:.2e}); {ratio:.2f}× the speed of FP64"
    data = {"photons": config["photons"], "processor": processor, "moved": float(moved), "noise": float(noise),
            "speed_ratio": ratio, "fp64_photons_per_second": fp64["PHOTONS_PER_SECOND"],
            "fp32_photons_per_second": fp32["PHOTONS_PER_SECOND"]}
    return outcome(None, headline, results, summaries=[fp64], data=data)


# ============================================================ the GPU vs the CPU, the same photons

def compare_gpu(config, processor, precision, folder):
    cpu_folder, gpu_folder = os.path.join(folder, "cpu"), os.path.join(folder, "gpu")
    cpu = run(config, cpu_folder, "cpu", precision)
    gpu = run(config, gpu_folder, "gpu", precision)
    compared = comparison(cpu_folder, gpu_folder)
    delete(cpu_folder, gpu_folder)
    differences = differing_counts(cpu, gpu)
    images = ("the images are identical, bit for bit" if compared["identical"] else
              f"{compared['differing_pixels']} pixels differ ({compared['moved']:.2e} of the photons moved)")
    results = ["The same seed: the same random numbers, so the same photons. A check of correctness, not of speed.",
               "", describe_star(config), "", *phases_table([("CPU (sequential)", cpu), ("GPU", gpu)]), "",
               "- counts: " + ("the same" if not differences else ", ".join(differences)),
               f"- images: {images}"]
    headline = f"the GPU vs the sequential CPU: {images}"
    data = {"photons": config["photons"], "precision": precision, "identical": compared["identical"],
            "differing_pixels": compared["differing_pixels"], "moved": compared["moved"],
            "same_counts": not differences}
    return outcome(compared["identical"] if precision == 64 else None, headline, results, summaries=[cpu, gpu],
                   data=data)


# ============================================================ running the checks

CHECKS = {"independent": compare_independent, "split": check_split, "passes": check_passes,
          "precision": compare_precision, "gpu": compare_gpu}


def run_check(name, config, processor, precision, folder):
    os.makedirs(folder, exist_ok=True)
    result = CHECKS[name](config, processor, precision, folder)
    return dict(result, name=name, processor=processor, precision=precision)
