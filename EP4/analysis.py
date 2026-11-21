import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
from PIL import Image as PilImage


def load_observer_image(path, frames, size):
    counts = np.fromfile(path, dtype=np.uint32)
    expected = frames * size * size
    if counts.size != expected:
        raise ValueError(f"{path}: expected {expected} counts, found {counts.size}")
    return counts.reshape(frames, size, size)


def load_ep2_image(path, nphi, observers, ny, nx):
    counts = np.fromfile(path, dtype=np.uint32)
    expected = nphi * observers * ny * nx
    if counts.size != expected:
        raise ValueError(f"{path}: expected {expected} counts, found {counts.size}")
    return counts.reshape(nphi, observers, ny, nx).transpose(1, 0, 2, 3)


def frame_to_picture(frame, brightest):
    normalized = np.clip(frame / brightest, 0, 1) if brightest > 0 else np.zeros_like(frame)
    rgb = (colormaps["hot"](normalized)[:, :, :3] * 255).astype(np.uint8)
    return PilImage.fromarray(np.flipud(rgb).copy(), mode="RGB")


def save_rotation_gif(frames, path, milliseconds_per_frame=80):
    brightest = np.percentile(frames, 99.9)
    pictures = [frame_to_picture(frame, brightest) for frame in frames]
    pictures[0].save(path, save_all=True, append_images=pictures[1:], duration=milliseconds_per_frame, loop=0)


def save_observer_pictures(image, names, folder):
    paths = []
    for frames, name in zip(image.astype(np.float64), names):
        gif_path = os.path.join(folder, f"{name}.gif")
        save_rotation_gif(frames, gif_path)
        frame_to_picture(frames[0], np.percentile(frames, 99.9)).save(os.path.join(folder, f"{name}_frame_0.png"))
        paths.append(gif_path)
    return paths


def chi2_between_independent_images(a, b, minimum_counts=20):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    total = a + b
    used = total >= minimum_counts
    residuals = (a[used] - b[used]) / np.sqrt(total[used])
    chi2 = float(np.sum(residuals ** 2))
    dof = int(used.sum())
    return {"chi2": chi2, "dof": dof, "chi2_per_dof": chi2 / dof,
            "sigma_from_1": (chi2 - dof) / np.sqrt(2 * dof), "residuals": residuals}


def fraction_of_photons_moved(a, b):
    changed_counts = np.abs(a.astype(np.int64) - b.astype(np.int64)).sum()
    return float(changed_counts / 2 / max(a.sum(), 1))


def relative_noise_per_lit_pixel(image):
    lit = image[image > 0]
    return float(1 / np.sqrt(lit.mean())) if lit.size else float("nan")


def plot_independent_comparison(a, b, chi2, labels, title, path):
    a_observer_0 = a[0].sum(axis=0).astype(np.float64)
    b_observer_0 = b[0].sum(axis=0).astype(np.float64)
    total = a_observer_0 + b_observer_0
    residual = np.divide(a_observer_0 - b_observer_0, np.sqrt(total), out=np.zeros_like(total), where=total > 0)
    brightest = np.percentile(np.concatenate([a_observer_0.ravel(), b_observer_0.ravel()]), 99.9)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for ax, picture, label in zip(axes[:2], (a_observer_0, b_observer_0), labels):
        shown = ax.imshow(picture, origin="lower", cmap="hot", vmin=0, vmax=brightest)
        ax.set_title(f"{label}\n(observer 0, all frames summed)")
        fig.colorbar(shown, ax=ax, label="photons per pixel")
    shown = axes[2].imshow(residual, origin="lower", cmap="RdBu_r", vmin=-4, vmax=4)
    axes[2].set_title("(a - b) / sqrt(a + b)")
    fig.colorbar(shown, ax=axes[2], label="sigma")
    x = np.linspace(-5, 5, 400)
    axes[3].hist(chi2["residuals"], bins=100, range=(-5, 5), density=True, alpha=0.7, label="pixels")
    axes[3].plot(x, np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi), "k--", label="standard normal")
    axes[3].set_title(f"residuals: chi2/dof = {chi2['chi2_per_dof']:.4f} ({chi2['dof']} pixels)")
    axes[3].set_xlabel("(a - b) / sqrt(a + b)")
    axes[3].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
