import argparse
import functools
import glob
import hashlib
import os
import shutil
import subprocess

import numpy as np

import analysis

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE_FOLDER = os.path.join(HERE, "engine")
BUILD_FOLDER = os.path.join(ENGINE_FOLDER, "build")
NVCC = "/usr/local/cuda/bin/nvcc"
SHARED_SOURCES = ["main.cu", "options.cu", "scene.cu", "observers.cu", "image.cu", "report.cu", "rng_test.cu"]
PROCESSOR_SOURCES = {"cpu": ["cpu.cu"],
                     "gpu": ["gpu.cu", "gpu_image.cu", "gpu_sms.cu", "gpu_device.cu", "gpu_rng_test.cu"]}
PROCESSORS = ("cpu", "gpu")
PRECISIONS = (64, 32)
FLAGS = ["-O2", "-std=c++17", "-Xcompiler", "-Wall"]
GPU_VARIANTS = {
    "sections": ["-DSTAR_SECTION_TIMERS"],
    "scene_in_global": ["-DSTAR_SCENE_IN_GLOBAL"],
    "registers_64": ["-maxrregcount=64"],
    "registers_40": ["-maxrregcount=40"],
    "no_fma": ["--fmad=false"],
}


# ============================================================ building

def binary_name(processor, precision, variant=None):
    return f"star_{processor}{precision}" + (f"_{variant}" if variant else "")


def binary_path(processor, precision, variant=None):
    return os.path.join(BUILD_FOLDER, binary_name(processor, precision, variant))


def variant_flags(processor, variant):
    if variant is None:
        return []
    if processor != "gpu" or variant not in GPU_VARIANTS:
        raise ValueError(f"no {processor} variant called {variant}; GPU variants: {', '.join(GPU_VARIANTS)}")
    return GPU_VARIANTS[variant]


def architecture_flags(architecture):
    if architecture == "native":
        return ["-arch=native"]
    return ["-gencode", f"arch=compute_{architecture},code=sm_{architecture}"]


def nvcc_command(processor, precision, architecture="native", variant=None):
    precision_flags = ["-DSTAR_FP32"] if precision == 32 else []
    report_flags = ["-Xptxas", "-v"] if processor == "gpu" else []
    sources = SHARED_SOURCES + PROCESSOR_SOURCES[processor]
    return [NVCC, *FLAGS, *architecture_flags(architecture), *precision_flags, *variant_flags(processor, variant),
            *report_flags, *sources, "-o", binary_path(processor, precision, variant)]


def engine_sources():
    return sorted(glob.glob(os.path.join(ENGINE_FOLDER, "*.cu")) + glob.glob(os.path.join(ENGINE_FOLDER, "*.cuh")))


@functools.cache
def code_hash():
    digest = hashlib.sha256()
    for path in engine_sources():
        digest.update(os.path.basename(path).encode())
        with open(path, "rb") as file:
            digest.update(file.read())
    return digest.hexdigest()[:12]


@functools.cache
def nvcc_version():
    return subprocess.run([NVCC, "--version"], capture_output=True, text=True).stdout


def build_hash(command):
    digest = hashlib.sha256()
    digest.update(code_hash().encode())
    digest.update(" ".join(command).encode())
    digest.update(nvcc_version().encode())
    return digest.hexdigest()[:12]


def read_text(path):
    if not os.path.exists(path):
        return None
    with open(path) as file:
        return file.read().strip()


def write_text(path, text):
    with open(path, "w") as file:
        file.write(text + "\n")


@functools.cache
def build_engine(processor="cpu", precision=64, architecture="native", verbose=False, variant=None):
    command = nvcc_command(processor, precision, architecture, variant)
    binary = binary_path(processor, precision, variant)
    wanted_hash = build_hash(command)
    if os.path.exists(binary) and read_text(binary + ".hash") == wanted_hash:
        return binary
    os.makedirs(BUILD_FOLDER, exist_ok=True)
    if verbose:
        print(" ".join(command))
    result = subprocess.run(command, cwd=ENGINE_FOLDER, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Error building {os.path.basename(binary)}:\n{result.stdout}{result.stderr}")
    write_text(binary + ".report", result.stderr)
    write_text(binary + ".hash", wanted_hash)
    return binary


# ============================================================ running

DEFAULT_OBSERVERS = [1.5708, 1.0472, 0.5236, 0.0]


def default_star_config(photons, **changes):
    config = {"photons": photons, "seed": 42, "limb": 1.0, "spots": 3, "observers": DEFAULT_OBSERVERS,
              "size": 200, "frames": 150, "nmu": None, "chunk_size": 1024, "observers_per_pass": None,
              "sms": None, "variant": None}
    config.update(changes)
    return config


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


def engine_command(config, output_dir, processor, precision, chunk_range):
    binary = build_engine(processor, precision, variant=config.get("variant"))
    command = [binary, "-n", config["photons"], "-s", config["seed"],
               "-a", config["limb"], "--spots", config["spots"],
               "--observers", ",".join(f"{i:.17g}" for i in config["observers"]),
               "--size", config["size"], "--frames", config["frames"],
               "--chunk-size", config["chunk_size"], "--output-dir", output_dir, "--code-hash", code_hash()]
    if config["nmu"] is not None:
        command += ["--nmu", config["nmu"]]
    if config["observers_per_pass"] is not None:
        command += ["--observers-per-pass", config["observers_per_pass"]]
    if config.get("sms") is not None:
        command += ["--sms", config["sms"]]
    if chunk_range is not None:
        command += ["--first-chunk", chunk_range[0], "--last-chunk", chunk_range[1]]
    return [str(item) for item in command]


def engine_output(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Error running the engine:\n{' '.join(command)}\n{result.stderr}")
    return result.stdout


def run_engine_output(config, output_dir, processor="cpu", precision=64, chunk_range=None):
    return engine_output(engine_command(config, output_dir, processor, precision, chunk_range))


def run_engine(config, output_dir, processor="cpu", precision=64, chunk_range=None):
    return lines_of_kind(run_engine_output(config, output_dir, processor, precision, chunk_range), "SUMMARY")[0]


def describe_engine(config, processor="cpu", precision=64):
    output = engine_output(engine_command(config, ".", processor, precision, None) + ["--describe"])
    return lines_of_kind(output, "CONFIG")[0], lines_of_kind(output, "OBSERVER")


# ============================================================ results

def read_record(path):
    with open(path) as file:
        return parse_key_values(file.read())


def observer_records(output_dir):
    records = (read_record(path) for path in glob.glob(os.path.join(output_dir, "*.record")))
    return sorted(records, key=lambda record: record["OBSERVER"])


def load_observer(output_dir, record):
    return analysis.load_observer_image(os.path.join(output_dir, record["IMAGE"]), record["FRAMES"], record["SIZE"])


def load_observers(output_dir):
    records = observer_records(output_dir)
    return records, np.stack([load_observer(output_dir, record) for record in records])


def load_and_delete_observers(output_dir):
    records, image = load_observers(output_dir)
    shutil.rmtree(output_dir)
    return records, image


# ============================================================ by hand

def main():
    parser = argparse.ArgumentParser(description="Build the EP4 engines")
    parser.add_argument("--arch", default="native", help="native, or a compute capability such as 80 (the A30)")
    args = parser.parse_args()
    for processor in PROCESSORS:
        for precision in PRECISIONS:
            print(build_engine(processor, precision, args.arch, verbose=True))


if __name__ == "__main__":
    main()
