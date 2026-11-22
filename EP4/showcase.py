import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime

import numpy as np
from PIL import Image as PilImage

import analysis
import engine
from gpu_monitor import GpuMonitor

RUNS_FOLDER = os.path.join(engine.HERE, "runs")
FFMPEG_NEXT_TO_PYTHON = os.path.join(os.path.dirname(sys.executable), "ffmpeg")
FFMPEG = shutil.which("ffmpeg") or (FFMPEG_NEXT_TO_PYTHON if os.path.exists(FFMPEG_NEXT_TO_PYTHON) else None)
FRAMES_PER_SECOND = 30
GIF_SIZE = 512
CENTER_RADIUS = 0.2
LIMB_RING = (0.90, 0.95)


# ============================================================ how many photons

def photons_for_snr(snr, size, frames, nmu, limb_darkening):
    return math.ceil(snr ** 2 * nmu * frames * (math.pi / 4) * size ** 2 * (1 + 2 * limb_darkening / 3))


def expected_counts(photons, size, frames, nmu, limb_darkening, mu):
    per_frame_and_bin = photons / (nmu * frames)
    return per_frame_and_bin * (4 / size ** 2) * (1 + limb_darkening * mu) / (math.pi * (1 + 2 * limb_darkening / 3))


# ============================================================ reading one observer without loading it all

def observer_frames(folder, record):
    path = os.path.join(folder, record["IMAGE"])
    return np.memmap(path, dtype=np.uint32, mode="r", shape=(record["FRAMES"], record["SIZE"], record["SIZE"]))


def disk_radius(size):
    centers = (np.arange(size) + 0.5) * 2 / size - 1
    x, y = np.meshgrid(centers, centers)
    return np.sqrt(x ** 2 + y ** 2)


def brightest_level(frames):
    sample = np.asarray(frames[::max(1, len(frames) // 30)], dtype=np.float32)
    return float(np.percentile(sample, 99.9))


def ring_counts(frames, radius, inner, outer, limb_darkening):
    ring = (radius >= inner) & (radius < outer)
    sample = np.asarray(frames[::max(1, len(frames) // 30)], dtype=np.float64)
    mu = np.sqrt(np.clip(1 - radius[ring] ** 2, 0, 1))
    return float(sample[:, ring].mean()), float(mu.mean())


# ============================================================ pictures and video

def ffmpeg_command(path, size):
    return [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{size}x{size}", "-r", str(FRAMES_PER_SECOND), "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", path]


def write_mp4(frames, brightest, path):
    size = frames.shape[1]
    video = subprocess.Popen(ffmpeg_command(path, size), stdin=subprocess.PIPE)
    for frame in frames:
        video.stdin.write(analysis.frame_to_picture(np.asarray(frame, dtype=np.float32), brightest).tobytes())
    video.stdin.close()
    video.wait()


def write_gif(frames, brightest, path):
    pictures = [analysis.frame_to_picture(np.asarray(frame, dtype=np.float32), brightest).resize((GIF_SIZE, GIF_SIZE))
                for frame in frames]
    pictures[0].save(path, save_all=True, append_images=pictures[1:], duration=1000 // FRAMES_PER_SECOND, loop=0)


def save_observer_media(folder, record, output_folder):
    frames = observer_frames(folder, record)
    brightest = brightest_level(frames)
    stem = record["IMAGE"].removesuffix(".bin")
    analysis.frame_to_picture(np.asarray(frames[0], dtype=np.float32), brightest).save(
        os.path.join(output_folder, stem + "_frame_0.png"))
    if FFMPEG:
        path = os.path.join(output_folder, stem + ".mp4")
        write_mp4(frames, brightest, path)
    else:
        path = os.path.join(output_folder, stem + ".gif")
        write_gif(frames, brightest, path)
    return path


# ============================================================ measured vs predicted SNR

def snr_check(folder, record, photons, nmu, limb_darkening):
    frames = observer_frames(folder, record)
    radius = disk_radius(record["SIZE"])
    rows = []
    for name, (inner, outer) in (("center", (0.0, CENTER_RADIUS)), ("limb", LIMB_RING)):
        measured, mu = ring_counts(frames, radius, inner, outer, limb_darkening)
        predicted = expected_counts(photons, record["SIZE"], record["FRAMES"], nmu, limb_darkening, mu)
        rows.append({"region": name, "mean_mu": mu, "counts_measured": measured, "counts_predicted": predicted,
                     "snr_measured": math.sqrt(measured), "snr_predicted": math.sqrt(predicted)})
    return rows


# ============================================================ the showcase

def parse_args():
    parser = argparse.ArgumentParser(description="EP4's showcase: a big run on the GPU, its video, its SNR")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--snr", type=float, default=10.0, help="the signal-to-noise ratio wanted at the limb")
    parser.add_argument("--precision", type=int, choices=engine.PRECISIONS, default=32)
    parser.add_argument("--plan", action="store_true", help="print N and the memory, run nothing")
    return parser.parse_args()


def main():
    args = parse_args()
    config = engine.default_star_config(1, size=args.size, frames=args.frames)
    described = engine.describe_engine(config, "gpu", args.precision)[0]
    nmu = described["NMU"]
    photons = photons_for_snr(args.snr, args.size, args.frames, nmu, config["limb"])
    config["photons"] = photons
    plan = engine.describe_engine(config, "gpu", args.precision)[0]
    print(f"{args.size}², {args.frames} frames, nmu {nmu}, SNR {args.snr} at the limb: N = {photons:.3e}; "
          f"{plan['OBSERVERS']} observers in {plan['PASSES']} pass(es), {plan['IMAGE_BYTES'] / 1e9:.2f} GB on the GPU",
          flush=True)
    if args.plan:
        return
    label = f"showcase-{args.size}-f{args.frames}-snr{args.snr:g}-gpu{args.precision}"
    folder = os.path.join(RUNS_FOLDER, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "-" + label)
    observers_folder = os.path.join(folder, "observers")
    os.makedirs(folder)
    with GpuMonitor(interval_ms=1000) as monitor:
        summary = engine.run_engine(config, observers_folder, "gpu", args.precision)
    print(f"photons {summary['PHOTONS_TIME']:.1f} s, whole run {summary['TIME']:.1f} s, "
          f"{summary['PHOTONS_PER_SECOND']:.3g} photons/s", flush=True)
    records = engine.observer_records(observers_folder)
    report = {"summary": summary, "gpu_monitor": monitor.summary(), "observers": []}
    for record in records:
        media = save_observer_media(observers_folder, record, folder)
        snr = snr_check(observers_folder, record, photons, nmu, config["limb"])
        report["observers"].append({"image": record["IMAGE"], "media": os.path.basename(media), "snr": snr})
        print(f"  {os.path.basename(media)}: SNR center {snr[0]['snr_measured']:.1f} "
              f"(predicted {snr[0]['snr_predicted']:.1f}), limb {snr[1]['snr_measured']:.1f} "
              f"(predicted {snr[1]['snr_predicted']:.1f})", flush=True)
    with open(os.path.join(folder, "showcase.json"), "w") as file:
        json.dump(report, file, indent=2)


if __name__ == "__main__":
    main()
