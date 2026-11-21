import argparse
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
SHARED_SOURCES = ["main.cu", "options.cu", "scene.cu", "observers.cu", "image.cu", "report.cu"]
PROCESSOR_SOURCES = {"cpu": ["cpu.cu"], "gpu": ["gpu.cu", "gpu_device.cu"]}
PROCESSORS = ("cpu", "gpu")
PRECISIONS = (64, 32)
FLAGS = ["-O2", "-std=c++17", "-Xcompiler", "-Wall"]


# ============================================================ building

def binary_name(processor, precision):
    return f"star_{processor}{precision}"


def binary_path(processor, precision):
    return os.path.join(BUILD_FOLDER, binary_name(processor, precision))


def architecture_flags(architecture):
    if architecture == "native":
        return ["-arch=native"]
    return ["-gencode", f"arch=compute_{architecture},code=sm_{architecture}"]


def nvcc_command(processor, precision, architecture="native"):
    precision_flags = ["-DSTAR_FP32"] if precision == 32 else []
    report_flags = ["-Xptxas", "-v"] if processor == "gpu" else []
    sources = SHARED_SOURCES + PROCESSOR_SOURCES[processor]
    return [NVCC, *FLAGS, *architecture_flags(architecture), *precision_flags, *report_flags,
            *sources, "-o", binary_path(processor, precision)]


def engine_sources():
    return sorted(glob.glob(os.path.join(ENGINE_FOLDER, "*.cu")) + glob.glob(os.path.join(ENGINE_FOLDER, "*.cuh")))


def code_hash():
    digest = hashlib.sha256()
    for path in engine_sources():
        digest.update(os.path.basename(path).encode())
        with open(path, "rb") as file:
            digest.update(file.read())
    return digest.hexdigest()[:12]


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


def build_engine(processor="cpu", precision=64, architecture="native", verbose=False):
    command = nvcc_command(processor, precision, architecture)
    binary = binary_path(processor, precision)
    wanted_hash = build_hash(command)
    if os.path.exists(binary) and read_text(binary + ".hash") == wanted_hash:
        return binary
    os.makedirs(BUILD_FOLDER, exist_ok=True)
    if verbose:
        print(" ".join(command))
    result = subprocess.run(command, cwd=ENGINE_FOLDER, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Error building {binary_name(processor, precision)}:\n{result.stdout}{result.stderr}")
    write_text(binary + ".report", result.stderr)
    write_text(binary + ".hash", wanted_hash)
    return binary


# ============================================================ running

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
    command = [build_engine(processor, precision), "-n", config["photons"], "-s", config["seed"],
               "-a", config["limb"], "--spots", config["spots"],
               "--observers", ",".join(f"{i:.17g}" for i in config["observers"]),
               "--size", config["size"], "--frames", config["frames"],
               "--chunk-size", config["chunk_size"], "--output-dir", output_dir, "--code-hash", code_hash()]
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


def run_engine(config, output_dir, processor="cpu", precision=64, chunk_range=None):
    output = engine_output(engine_command(config, output_dir, processor, precision, chunk_range))
    return lines_of_kind(output, "SUMMARY")[0]


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
