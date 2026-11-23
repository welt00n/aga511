import math

from lab import snr

BYTES_PER_COUNT = 4
LIMB_DARKENING = 1.0


# ============================================================ resolution: the bins, and the blur they make

def nmu_for(frames):
    return max(1, 2 * round((frames / math.pi - 1) / 2) + 1)


def rotation_blur(size, frames):
    return size / 2 * 2 * math.pi / frames


def inclination_bin(inclination, nmu):
    mu = math.cos(inclination)
    index = min(int((mu + 1) / 2 * nmu), nmu - 1)
    mu_low, mu_high = -1 + 2 * index / nmu, -1 + 2 * (index + 1) / nmu
    return math.acos((mu_low + mu_high) / 2), math.acos(mu_high), math.acos(mu_low)


def tilt_blur(size, inclination, nmu):
    _, low, high = inclination_bin(inclination, nmu)
    return size / 2 * (high - low)


# ============================================================ the two costs

def photons(size, frames, nmu, wanted_snr):
    return snr.photons_for_snr(wanted_snr, size, frames, nmu, LIMB_DARKENING)


def image_bytes(size, frames, observers):
    return BYTES_PER_COUNT * observers * frames * size ** 2


def passes(size, frames, observers, memory):
    per_observer = image_bytes(size, frames, 1)
    fit = int(memory // per_observer)
    return math.ceil(observers / fit) if fit else None


def configuration(size, frames, observers, wanted_snr, speed, memory):
    nmu = nmu_for(frames)
    n = photons(size, frames, nmu, wanted_snr)
    runs = passes(size, frames, observers, memory)
    return {"size": size, "frames": frames, "nmu": nmu, "observers": observers, "snr": wanted_snr,
            "photons": n, "passes": runs, "seconds": runs * n / speed if runs else None,
            "bytes": image_bytes(size, frames, observers), "bytes_per_observer": image_bytes(size, frames, 1),
            "rotation_blur_px": rotation_blur(size, frames), "equator_blur_px": tilt_blur(size, math.pi / 2, nmu),
            "pole_blur_px": tilt_blur(size, 0.0, nmu), "video_seconds_at_30fps": frames / 30}
