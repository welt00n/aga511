import argparse
import json
import os
import subprocess
from datetime import datetime

import numpy as np

import analysis

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE_FOLDER = os.path.join(HERE, "engine")
EP2_ENGINE_FOLDER = os.path.join(HERE, "..", "EP2", "engine")
RUNS_FOLDER = os.path.join(HERE, "runs")


# ============================================================ command line

def photons(text):
    value = float(text)
    if value < 1 or value != int(value):
        raise argparse.ArgumentTypeError("the number of photons must be a whole number >= 1")
    return int(value)


def parse_args():
    parser = argparse.ArgumentParser(description="EP4: Monte Carlo image of a spotted star")
    parser.add_argument("-n", "--photons", type=photons, default=10 ** 7, help="number of photons, e.g. 1e8")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limb", type=float, default=1.0, help="limb darkening a in I = I0 (1 + a mu)")
    parser.add_argument("--spots", type=int, default=3)
    parser.add_argument("--observers", type=int, default=4)
    parser.add_argument("--nx", type=int, default=200)
    parser.add_argument("--ny", type=int, default=200)
    parser.add_argument("--nphi", type=int, default=150)
    parser.add_argument("--nmu", type=int, default=150)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--precision", type=int, choices=(64, 32), default=64)
    parser.add_argument("--compare-ep2", action="store_true", help="EP4 vs EP2, both without spots")
    parser.add_argument("--check-split", action="store_true", help="two halves of the chunk range vs the full run")
    parser.add_argument("--compare-precision", action="store_true", help="FP32 vs FP64 on the same photons")
    return parser.parse_args()


def star_config(args, **changes):
    config = {"photons": args.photons, "seed": args.seed, "limb": args.limb, "spots": args.spots,
              "observers": args.observers, "nx": args.nx, "ny": args.ny, "nphi": args.nphi, "nmu": args.nmu,
              "chunk_size": args.chunk_size}
    config.update(changes)
    return config


# ============================================================ engine

def compile_engine():
    result = subprocess.run(["make", "-s"], cwd=ENGINE_FOLDER, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Error compiling the engine:\n{result.stdout}{result.stderr}")


def parse_key_values(line):
    values = {}
    for pair in line.split():
        key, text = pair.split("=", 1)
        for convert in (int, float, str):
            try:
                values[key] = convert(text)
                break
            except ValueError:
                pass
    return values


def run_engine(config, image_path, precision=64, chunk_range=None):
    engine = os.path.join(ENGINE_FOLDER, "star" if precision == 64 else "star32")
    command = [engine, "-n", config["photons"], "-s", config["seed"], "-a", config["limb"],
               "--spots", config["spots"], "--observers", config["observers"],
               "--nx", config["nx"], "--ny", config["ny"], "--nphi", config["nphi"], "--nmu", config["nmu"],
               "--chunk-size", config["chunk_size"], "-o", image_path]
    if chunk_range is not None:
        command += ["--first-chunk", chunk_range[0], "--last-chunk", chunk_range[1]]
    command = [str(item) for item in command]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Error running the engine:\n{' '.join(command)}\n{result.stderr}")
    return parse_key_values(result.stdout.strip().splitlines()[-1])


def load_and_delete_image(result):
    image = analysis.load_image(result["IMAGE"], result["NPHI"], result["OBSERVERS"], result["NY"], result["NX"])
    os.remove(result["IMAGE"])
    return image


def compile_and_run_ep2(config, folder):
    binary = os.path.join(folder, "ep2_star")
    sources = [os.path.join(EP2_ENGINE_FOLDER, name) for name in ("main.c", "star.c", "base.c")]
    compiled = subprocess.run(["gcc", *sources, "-o", binary, "-O2", "-lm"], capture_output=True, text=True)
    if compiled.returncode != 0:
        raise RuntimeError(f"Error compiling EP2:\n{compiled.stderr}")
    arguments = [config[key] for key in ("photons", "seed", "limb", "spots", "observers", "nx", "ny", "nphi", "nmu")]
    started_at = datetime.now()
    ran = subprocess.run([binary, *map(str, arguments)], cwd=folder, capture_output=True, text=True)
    seconds = (datetime.now() - started_at).total_seconds()
    os.remove(binary)
    if ran.returncode != 0:
        raise RuntimeError(f"Error running EP2:\n{ran.stderr}")
    image_name = next(name for name in os.listdir(folder) if name.startswith("star-N-") and name.endswith(".bin"))
    image_path = os.path.join(folder, image_name)
    image = analysis.load_image(image_path, config["nphi"], config["observers"], config["ny"], config["nx"])
    os.remove(image_path)
    return image, seconds


# ============================================================ output

def new_run_folder(name):
    folder = os.path.join(RUNS_FOLDER, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "-" + name)
    os.makedirs(folder)
    return folder


def save_json(data, path):
    with open(path, "w") as file:
        json.dump(data, file, indent=2)


def save_report(lines, path):
    text = "\n".join(lines)
    print(text)
    with open(path, "w") as file:
        file.write(text + "\n")


def describe_run(result):
    return (f"{result['PHOTONS']:.3g} photons in {result['PHOTONS_TIME']:.3f} s "
            f"({result['PHOTONS_PER_SECOND']:.3g} photons/s); setup {result['SETUP_TIME']:.3f} s, "
            f"save {result['SAVE_TIME']:.3f} s; recorded {result['RECORDED']} "
            f"({100 * result['RECORDED'] / result['PHOTONS']:.2f}%), blocked by spots {result['BLOCKED']}")


def describe_star(config):
    return (f"N = {config['photons']:.3g}, {config['nx']}x{config['ny']} pixels, nphi = {config['nphi']}, "
            f"nmu = {config['nmu']}, {config['observers']} observers, {config['spots']} spots")


# ============================================================ experiments

def simulate_star(args):
    config = star_config(args)
    folder = new_run_folder(f"cpu-fp{args.precision}")
    result = run_engine(config, os.path.join(folder, "image.bin"), args.precision)
    save_json({"config": config, "result": result}, os.path.join(folder, "run.json"))
    print(describe_run(result))
    for path in analysis.save_observer_pictures(load_and_delete_image(result), folder):
        print(f"  {os.path.relpath(path, HERE)}")


def compare_with_ep2(args):
    config = star_config(args, spots=0)
    folder = new_run_folder(f"compare-ep2-fp{args.precision}")
    ep4 = run_engine(config, os.path.join(folder, "ep4.bin"), args.precision)
    ep4_image = load_and_delete_image(ep4)
    ep2_image, ep2_seconds = compile_and_run_ep2(config, folder)
    chi2 = analysis.chi2_between_independent_images(ep4_image.sum(axis=0), ep2_image.sum(axis=0))
    verdict = "the same physics within noise" if abs(chi2["sigma_from_1"]) < 3 else "THE IMAGES DIFFER"
    lines = ["EP4 vs EP2, without spots, independent random numbers (frames summed)",
             f"  {describe_star(config)}",
             f"  EP4: {describe_run(ep4)}",
             f"  EP2: {config['photons']:.3g} photons in {ep2_seconds:.3f} s "
             f"({config['photons'] / ep2_seconds:.3g} photons/s, including its setup and save)",
             f"  chi2/dof = {chi2['chi2_per_dof']:.4f} over {chi2['dof']} pixels "
             f"({chi2['sigma_from_1']:+.2f} sigma from 1): {verdict}",
             "  photons per observer, EP4 vs EP2:"]
    for observer, (a, b) in enumerate(zip(ep4_image.sum(axis=(0, 2, 3)), ep2_image.sum(axis=(0, 2, 3)))):
        lines.append(f"    observer {observer}: {a} vs {b} ({(int(a) - int(b)) / np.sqrt(a + b):+.2f} sigma)")
    save_report(lines, os.path.join(folder, "report.txt"))
    analysis.plot_independent_comparison(ep4_image, ep2_image, chi2, ("EP4", "EP2"),
                                         "EP4 vs EP2 (no spots, independent random numbers)",
                                         os.path.join(folder, "comparison.png"))


def check_split(args):
    config = star_config(args)
    folder = new_run_folder(f"check-split-fp{args.precision}")
    full = run_engine(config, os.path.join(folder, "full.bin"), args.precision)
    chunks = full["CHUNKS"]
    middle = chunks // 2
    first = run_engine(config, os.path.join(folder, "first.bin"), args.precision, (0, middle))
    second = run_engine(config, os.path.join(folder, "second.bin"), args.precision, (middle, chunks))
    halves = load_and_delete_image(first).astype(np.uint64) + load_and_delete_image(second)
    identical = np.array_equal(halves, load_and_delete_image(full))
    save_report([f"Chunks [0, {middle}) + [{middle}, {chunks}) vs the full run [0, {chunks})",
                 f"  {describe_star(config)}, chunk size {config['chunk_size']}",
                 f"  recorded: {first['RECORDED']} + {second['RECORDED']} = "
                 f"{first['RECORDED'] + second['RECORDED']} vs {full['RECORDED']}",
                 "  images: " + ("IDENTICAL, bit for bit" if identical else "DIFFERENT")],
                os.path.join(folder, "report.txt"))
    if not identical:
        raise SystemExit(1)


def compare_precision(args):
    config = star_config(args)
    folder = new_run_folder("compare-precision")
    fp64 = run_engine(config, os.path.join(folder, "fp64.bin"), 64)
    fp32 = run_engine(config, os.path.join(folder, "fp32.bin"), 32)
    fp64_image = load_and_delete_image(fp64)
    fp32_image = load_and_delete_image(fp32)
    moved = analysis.fraction_of_photons_moved(fp64_image, fp32_image)
    noise = analysis.relative_noise_per_lit_pixel(fp64_image.sum(axis=0))
    save_report(["FP32 vs FP64, same seed: the same random numbers, so the same photons",
                 f"  {describe_star(config)}",
                 f"  FP64: {describe_run(fp64)}",
                 f"  FP32: {describe_run(fp32)}",
                 f"  speed FP32 / FP64: {fp32['PHOTONS_PER_SECOND'] / fp64['PHOTONS_PER_SECOND']:.2f}",
                 f"  photons that land in another pixel in FP32: {moved:.2e}",
                 f"  Poisson noise per lit pixel (frames summed): {noise:.2e}"],
                os.path.join(folder, "report.txt"))


def main():
    args = parse_args()
    compile_engine()
    checks = [(args.compare_ep2, compare_with_ep2), (args.check_split, check_split),
              (args.compare_precision, compare_precision)]
    requested = [check for wanted, check in checks if wanted]
    for experiment in requested or [simulate_star]:
        experiment(args)


if __name__ == "__main__":
    main()
