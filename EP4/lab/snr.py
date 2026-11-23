import math

import numpy as np

from lab.videos import observer_frames

CENTER_RADIUS = 0.2
LIMB_RING = (0.90, 0.95)


# ============================================================ how many photons

def photons_for_snr(snr, size, frames, nmu, limb_darkening):
    return math.ceil(snr ** 2 * nmu * frames * (math.pi / 4) * size ** 2 * (1 + 2 * limb_darkening / 3))


def expected_counts(photons, size, frames, nmu, limb_darkening, mu):
    per_frame_and_bin = photons / (nmu * frames)
    return per_frame_and_bin * (4 / size ** 2) * (1 + limb_darkening * mu) / (math.pi * (1 + 2 * limb_darkening / 3))


# ============================================================ measured vs predicted SNR

def disk_radius(size):
    centers = (np.arange(size) + 0.5) * 2 / size - 1
    x, y = np.meshgrid(centers, centers)
    return np.sqrt(x ** 2 + y ** 2)


def ring_counts(frames, radius, inner, outer):
    ring = (radius >= inner) & (radius < outer)
    sample = np.asarray(frames[::max(1, len(frames) // 30)], dtype=np.float64)
    mu = np.sqrt(np.clip(1 - radius[ring] ** 2, 0, 1))
    return float(sample[:, ring].mean()), float(mu.mean())


def snr_check(folder, record, photons, nmu, limb_darkening):
    frames = observer_frames(folder, record)
    radius = disk_radius(record["SIZE"])
    rows = []
    for name, (inner, outer) in (("center", (0.0, CENTER_RADIUS)), ("limb", LIMB_RING)):
        measured, mu = ring_counts(frames, radius, inner, outer)
        predicted = expected_counts(photons, record["SIZE"], record["FRAMES"], nmu, limb_darkening, mu)
        rows.append({"region": name, "mean_mu": mu, "counts_measured": measured, "counts_predicted": predicted,
                     "snr_measured": math.sqrt(measured), "snr_predicted": math.sqrt(predicted)})
    return rows


# ============================================================ the SNR, as a table and a headline

def snr_table(observers):
    lines = ["| observer | video | SNR center: measured / predicted | SNR limb ring: measured / predicted |",
             "|---|---|---|---|"]
    for observer in observers:
        center, limb = observer["snr"]
        lines.append(f"| {observer['image'].removesuffix('.bin')} | [{observer['media']}]({observer['media']}) | "
                     f"{center['snr_measured']:.1f} / {center['snr_predicted']:.1f} | "
                     f"{limb['snr_measured']:.1f} / {limb['snr_predicted']:.1f} |")
    return lines


def snr_headline(summary, observers):
    centers = [observer["snr"][0] for observer in observers]
    limbs = [observer["snr"][1] for observer in observers]
    return (f"N = {summary['PHOTONS']:.2e} photons in {summary['PHOTONS_TIME']:.3g} s "
            f"({summary['PHOTONS_PER_SECOND']:.2g} photons/s); SNR center {np.median([c['snr_measured'] for c in centers]):.1f} "
            f"(predicted {centers[0]['snr_predicted']:.1f}), limb {np.median([l['snr_measured'] for l in limbs]):.1f} "
            f"(predicted {limbs[0]['snr_predicted']:.1f})")
