import functools
import glob
import hashlib
import os
import re
import shutil
import subprocess

import numpy as np

from lab import analysis, speeds

EP4_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE_FOLDER = os.path.join(EP4_FOLDER, "engine")
BUILD_FOLDER = os.path.join(ENGINE_FOLDER, "build")
DEFAULT_CUDA = "/usr/local/cuda"


def cuda_tool(name):
    for home in (DEFAULT_CUDA, os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH")):
        if home and os.path.exists(os.path.join(home, "bin", name)):
            return os.path.join(home, "bin", name)
    return shutil.which(name) or os.path.join(DEFAULT_CUDA, "bin", name)


NVCC = cuda_tool("nvcc")
SHARED_SOURCES = ["main.cu", "run/options.cu", "physics/scene.cu", "physics/observers.cu", "run/image.cu",
                  "run/report.cu", "random/rng_test.cu"]
PROCESSOR_SOURCES = {"cpu": ["cpu/cpu.cu"],
                     "gpu": ["gpu/gpu.cu", "gpu/gpu_image.cu", "gpu/gpu_sms.cu", "gpu/gpu_device.cu",
                             "random/gpu_rng_test.cu"]}
PROCESSORS = ("cpu", "gpu")
PRECISIONS = (64, 32)
FLAGS = ["-O2", "-std=c++17", "-I.", "-Xcompiler", "-Wall"]
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


NATIVE_SINCE = (11, 6)


@functools.lru_cache(maxsize=None)
def nvcc_release():
    found = re.search(r"release (\d+)\.(\d+)", nvcc_version())
    return (int(found.group(1)), int(found.group(2))) if found else (0, 0)


@functools.lru_cache(maxsize=None)
def gpu_compute_capability():
    result = subprocess.run([shutil.which("nvidia-smi") or "nvidia-smi", "--query-gpu=compute_cap",
                             "--format=csv,noheader"], capture_output=True, text=True)
    first = result.stdout.split()[0] if result.returncode == 0 and result.stdout.split() else ""
    return first.replace(".", "") if re.fullmatch(r"\d+\.\d+", first) else None


def architecture_flags(architecture):
    if architecture == "native" and nvcc_release() < NATIVE_SINCE and gpu_compute_capability():
        architecture = gpu_compute_capability()
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
    paths = glob.glob(os.path.join(ENGINE_FOLDER, "**", "*.cu*"), recursive=True)
    return sorted(path for path in paths if not path.startswith(BUILD_FOLDER + os.sep))


@functools.cache
def code_hash():
    digest = hashlib.sha256()
    for path in engine_sources():
        digest.update(os.path.relpath(path, ENGINE_FOLDER).encode())
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


def build_all(architecture="native"):
    return [build_engine(processor, precision, architecture, verbose=True)
            for processor in PROCESSORS for precision in PRECISIONS]


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
    output = engine_output(engine_command(config, output_dir, processor, precision, chunk_range))
    if config.get("variant") is None:
        for summary in lines_of_kind(output, "SUMMARY"):
            speeds.remember_engine_run(summary)
    return output


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


def delete_observer_images(output_dir):
    for record in observer_records(output_dir):
        os.remove(os.path.join(output_dir, record["IMAGE"]))
