import argparse
import glob
import json
import math
import os
import shutil
import subprocess
from datetime import datetime

import numpy as np

import analysis

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE_FOLDER = os.path.join(HERE, "engine")
EP2_ENGINE_FOLDER = os.path.join(HERE, "..", "EP2", "engine")
RUNS_FOLDER = os.path.join(HERE, "runs")
DEFAULT_OBSERVERS = [1.5708, 1.0472, 0.5236, 0.0]


# ============================================================ command line

def photons(text):
    value = float(text)
    if value < 1 or value != int(value):
        raise argparse.ArgumentTypeError("the number of photons must be a whole number >= 1")
    return int(value)


def inclinations(text):
    return [float(inclination) for inclination in text.split(",")]


def parse_args():
    parser = argparse.ArgumentParser(description="EP4: Monte Carlo image of a spotted star")
    parser.add_argument("-n", "--photons", type=photons, default=10 ** 7, help="number of photons, e.g. 1e8")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limb", type=float, default=1.0, help="limb darkening a in I = I0 (1 + a mu)")
    parser.add_argument("--spots", type=int, default=3)
    parser.add_argument("--observers", type=inclinations, default=DEFAULT_OBSERVERS,
                        help="inclinations in radians, e.g. 1.5708,0.7854,0")
    parser.add_argument("--size", type=int, default=200, help="square image side in pixels")
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--nmu", type=int, default=None, help="mu bins (default: the odd number nearest frames/pi)")
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--observers-per-pass", type=int, default=None)
    parser.add_argument("--precision", type=int, choices=(64, 32), default=64)
    parser.add_argument("--compare-ep2", action="store_true", help="EP4 vs EP2, both without spots")
    parser.add_argument("--check-split", action="store_true", help="two halves of the chunk range vs the full run")
    parser.add_argument("--check-passes", action="store_true", help="one observer per pass vs all in one pass")
    parser.add_argument("--compare-precision", action="store_true", help="FP32 vs FP64 on the same photons")
    return parser.parse_args()


def star_config(args, **changes):
    config = {"photons": args.photons, "seed": args.seed, "limb": args.limb, "spots": args.spots,
              "observers": args.observers, "size": args.size, "frames": args.frames, "nmu": args.nmu,
              "chunk_size": args.chunk_size, "observers_per_pass": args.observers_per_pass}
    config.update(changes)
    return config


# ============================================================ engine

def compile_engine():
    result = subprocess.run(["make", "-s"], cwd=ENGINE_FOLDER, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Error compiling the engine:\n{result.stdout}{result.stderr}")


def parse_key_values(text):
    values = {}
    for pair in text.split():
        if "=" not in pair:
            continue
        key, value = pair.split("=", 1)
        for convert in (int, float, str):
            try:
                values[key] = convert(value)
                break
            except ValueError:
                pass
    return values


def lines_of_kind(output, kind):
    return [parse_key_values(line) for line in output.splitlines() if line.startswith(kind + " ")]


def engine_command(config, output_dir, precision, chunk_range):
    engine = os.path.join(ENGINE_FOLDER, "star" if precision == 64 else "star32")
    command = [engine, "-n", config["photons"], "-s", config["seed"], "-a", config["limb"],
               "--spots", config["spots"], "--observers", ",".join(f"{i:.17g}" for i in config["observers"]),
               "--size", config["size"], "--frames", config["frames"],
               "--chunk-size", config["chunk_size"], "--output-dir", output_dir]
    if config["nmu"] is not None:
        command += ["--nmu", config["nmu"]]
    if config["observers_per_pass"] is not None:
        command += ["--observers-per-pass", config["observers_per_pass"]]
    if chunk_range is not None:
        command += ["--first-chunk", chunk_range[0], "--last-chunk", chunk_range[1]]
    return [str(item) for item in command]


def engine_output(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Error running the engine:\n{' '.join(command)}\n{result.stderr}")
    return result.stdout


def run_engine(config, output_dir, precision=64, chunk_range=None):
    output = engine_output(engine_command(config, output_dir, precision, chunk_range))
    return lines_of_kind(output, "SUMMARY")[0]


def describe_engine(config, precision=64):
    output = engine_output(engine_command(config, ".", precision, None) + ["--describe"])
    return lines_of_kind(output, "CONFIG")[0], lines_of_kind(output, "OBSERVER")


def read_record(path):
    with open(path) as file:
        return parse_key_values(file.read())


def load_observers(output_dir):
    records = sorted((read_record(path) for path in glob.glob(os.path.join(output_dir, "*.record"))),
                     key=lambda record: record["OBSERVER"])
    image = np.stack([analysis.load_observer_image(os.path.join(output_dir, record["IMAGE"]),
                                                   record["FRAMES"], record["SIZE"]) for record in records])
    return records, image


def load_and_delete_observers(output_dir):
    records, image = load_observers(output_dir)
    shutil.rmtree(output_dir)
    return records, image


def ep2_observer_inclinations(count, nmu):
    bins = [nmu // 2] if count == 1 else [observer * (nmu - 1) // (count - 1) for observer in range(count)]
    return [math.acos(-1 + (bin + 0.5) * 2 / nmu) for bin in bins]


def compile_and_run_ep2(config, folder):
    binary = os.path.join(folder, "ep2_star")
    sources = [os.path.join(EP2_ENGINE_FOLDER, name) for name in ("main.c", "star.c", "base.c")]
    compiled = subprocess.run(["gcc", *sources, "-o", binary, "-O2", "-lm"], capture_output=True, text=True)
    if compiled.returncode != 0:
        raise RuntimeError(f"Error compiling EP2:\n{compiled.stderr}")
    observers = len(config["observers"])
    arguments = [config["photons"], config["seed"], config["limb"], config["spots"], observers,
                 config["size"], config["size"], config["frames"], config["nmu"]]
    started_at = datetime.now()
    ran = subprocess.run([binary, *map(str, arguments)], cwd=folder, capture_output=True, text=True)
    seconds = (datetime.now() - started_at).total_seconds()
    os.remove(binary)
    if ran.returncode != 0:
        raise RuntimeError(f"Error running EP2:\n{ran.stderr}")
    image_name = next(name for name in os.listdir(folder) if name.startswith("star-N-") and name.endswith(".bin"))
    image_path = os.path.join(folder, image_name)
    image = analysis.load_ep2_image(image_path, config["frames"], observers, config["size"], config["size"])
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
    return (f"{result['PHOTONS']:.3g} photons, {result['PASSES']} pass(es), in {result['PHOTONS_TIME']:.3f} s "
            f"({result['PHOTONS_PER_SECOND']:.3g} photons/s); setup {result['SETUP_TIME']:.3f} s, "
            f"zero {result['ZERO_TIME']:.3f} s, save {result['SAVE_TIME']:.3f} s; recorded {result['RECORDED']} "
            f"({100 * result['RECORDED'] / result['PHOTONS']:.2f}%), blocked by spots {result['BLOCKED']}")


def describe_star(config):
    nmu = config["nmu"] if config["nmu"] is not None else "odd nearest frames/pi"
    observers = ", ".join(f"{i:g}" for i in config["observers"])
    return (f"N = {config['photons']:.3g}, {config['size']}x{config['size']} pixels, {config['frames']} frames, "
            f"nmu = {nmu}, observers at {observers} rad, {config['spots']} spots")


# ============================================================ experiments

def simulate_star(args):
    config = star_config(args)
    folder = new_run_folder(f"cpu-fp{args.precision}")
    observers_folder = os.path.join(folder, "observers")
    result = run_engine(config, observers_folder, args.precision)
    save_json({"config": config, "result": result}, os.path.join(folder, "run.json"))
    print(describe_run(result))
    records, image = load_observers(observers_folder)
    names = [record["IMAGE"].removesuffix(".bin") for record in records]
    for path in analysis.save_observer_pictures(image, names, folder):
        print(f"  {os.path.relpath(path, HERE)}")
    for record in records:
        os.remove(os.path.join(observers_folder, record["IMAGE"]))


def compare_with_ep2(args):
    nmu = describe_engine(star_config(args), args.precision)[0]["NMU"]
    config = star_config(args, spots=0, nmu=nmu, observers=ep2_observer_inclinations(len(args.observers), nmu))
    folder = new_run_folder(f"compare-ep2-fp{args.precision}")
    ep4 = run_engine(config, os.path.join(folder, "ep4"), args.precision)
    _, ep4_image = load_and_delete_observers(os.path.join(folder, "ep4"))
    ep2_image, ep2_seconds = compile_and_run_ep2(config, folder)
    chi2 = analysis.chi2_between_independent_images(ep4_image.sum(axis=1), ep2_image.sum(axis=1))
    verdict = "the same physics within noise" if abs(chi2["sigma_from_1"]) < 3 else "THE IMAGES DIFFER"
    lines = ["EP4 vs EP2, without spots, independent random numbers (frames summed)",
             f"  {describe_star(config)}",
             f"  EP4: {describe_run(ep4)}",
             f"  EP2: {config['photons']:.3g} photons in {ep2_seconds:.3f} s "
             f"({config['photons'] / ep2_seconds:.3g} photons/s, including its setup and save)",
             f"  chi2/dof = {chi2['chi2_per_dof']:.4f} over {chi2['dof']} pixels "
             f"({chi2['sigma_from_1']:+.2f} sigma from 1): {verdict}",
             "  photons per observer, EP4 vs EP2:"]
    for observer, (a, b) in enumerate(zip(ep4_image.sum(axis=(1, 2, 3)), ep2_image.sum(axis=(1, 2, 3)))):
        lines.append(f"    observer {observer}: {a} vs {b} ({(int(a) - int(b)) / np.sqrt(a + b):+.2f} sigma)")
    save_report(lines, os.path.join(folder, "report.txt"))
    analysis.plot_independent_comparison(ep4_image, ep2_image, chi2, ("EP4", "EP2"),
                                         "EP4 vs EP2 (no spots, independent random numbers)",
                                         os.path.join(folder, "comparison.png"))


def check_split(args):
    config = star_config(args)
    folder = new_run_folder(f"check-split-fp{args.precision}")
    full = run_engine(config, os.path.join(folder, "full"), args.precision)
    chunks = full["CHUNKS"]
    middle = chunks // 2
    first = run_engine(config, os.path.join(folder, "first"), args.precision, (0, middle))
    second = run_engine(config, os.path.join(folder, "second"), args.precision, (middle, chunks))
    halves = (load_and_delete_observers(os.path.join(folder, "first"))[1].astype(np.uint64)
              + load_and_delete_observers(os.path.join(folder, "second"))[1])
    identical = np.array_equal(halves, load_and_delete_observers(os.path.join(folder, "full"))[1])
    save_report([f"Chunks [0, {middle}) + [{middle}, {chunks}) vs the full run [0, {chunks})",
                 f"  {describe_star(config)}, chunk size {config['chunk_size']}",
                 f"  recorded: {first['RECORDED']} + {second['RECORDED']} = "
                 f"{first['RECORDED'] + second['RECORDED']} vs {full['RECORDED']}",
                 "  images: " + ("IDENTICAL, bit for bit" if identical else "DIFFERENT")],
                os.path.join(folder, "report.txt"))
    if not identical:
        raise SystemExit(1)


def check_passes(args):
    folder = new_run_folder(f"check-passes-fp{args.precision}")
    one_pass = run_engine(star_config(args, observers_per_pass=None), os.path.join(folder, "one-pass"), args.precision)
    passes = run_engine(star_config(args, observers_per_pass=1), os.path.join(folder, "passes"), args.precision)
    one_pass_image = load_and_delete_observers(os.path.join(folder, "one-pass"))[1]
    passes_image = load_and_delete_observers(os.path.join(folder, "passes"))[1]
    identical = np.array_equal(one_pass_image, passes_image)
    same_counts = all(one_pass[key] == passes[key] for key in ("PHOTONS", "BLOCKED", "MISSED", "RECORDED"))
    save_report([f"{one_pass['OBSERVERS']} observers in 1 pass vs {passes['PASSES']} passes of 1 observer",
                 f"  {describe_star(star_config(args))}",
                 f"  1 pass:   {describe_run(one_pass)}",
                 f"  {passes['PASSES']} passes: {describe_run(passes)}",
                 "  counts: " + ("the same" if same_counts else "DIFFERENT"),
                 "  images: " + ("IDENTICAL, bit for bit" if identical else "DIFFERENT")],
                os.path.join(folder, "report.txt"))
    if not (identical and same_counts):
        raise SystemExit(1)


def compare_precision(args):
    config = star_config(args)
    folder = new_run_folder("compare-precision")
    fp64 = run_engine(config, os.path.join(folder, "fp64"), 64)
    fp32 = run_engine(config, os.path.join(folder, "fp32"), 32)
    fp64_image = load_and_delete_observers(os.path.join(folder, "fp64"))[1]
    fp32_image = load_and_delete_observers(os.path.join(folder, "fp32"))[1]
    moved = analysis.fraction_of_photons_moved(fp64_image, fp32_image)
    noise = analysis.relative_noise_per_lit_pixel(fp64_image.sum(axis=1))
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
              (args.check_passes, check_passes), (args.compare_precision, compare_precision)]
    requested = [check for wanted, check in checks if wanted]
    for experiment in requested or [simulate_star]:
        experiment(args)


if __name__ == "__main__":
    main()
