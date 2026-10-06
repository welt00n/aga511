import os
from datetime import datetime
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib import colormaps

def save_frames(frames, output_path):
	filepath = os.path.join(output_path, "frames.npy")
	np.save(filepath, frames)

def frame_to_image(frame, vmax):
    normalized = frame / vmax
    rgba = colormaps["hot"](normalized)
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8).copy()
    return Image.fromarray(rgb, mode="RGB")

def save_pics(frames, vmax, output_path):
    for i in np.linspace(0, frames.shape[0]-1, 10, dtype=int):
        image = frame_to_image(frames[i], vmax)
        filepath = os.path.join(output_path, f"frame_{i:04d}.png")
        image.save(filepath)

def save_animation(frames, vmax, output_path):
    images = [frame_to_image(frame, vmax) for frame in frames]
    filepath = os.path.join(output_path, "animation.gif")
    images[0].save(
        filepath,
        save_all=True,
        append_images=images[1:],
        duration=123,  # milliseconds per frame
        loop=0
    )

def make_analysis(image, config, run_id):
	vmax = image.max()
	n_path = os.path.join(run_id, str(config['N']))
	os.makedirs(n_path, exist_ok=True)
	for angle in range(config['n_observers']):
		angle_output_path = os.path.join(n_path, f"Observer {angle}")
		os.makedirs(angle_output_path, exist_ok=True)
		frames = image[:, angle].astype(np.float64)
		save_frames(frames, angle_output_path)
		save_pics(frames, vmax, angle_output_path)
		save_animation(frames, vmax, angle_output_path)

def save_time_runtime_graph(data, output_path):
    N = np.array([point["N"] for point in data], dtype=float)
    T = np.array([point["T"] for point in data], dtype=float)

    # ---------------------------------------------------------
    # Fit T = c * N^p
    # Use only the large-N region, where fixed overhead matters less.
    # ---------------------------------------------------------
    fit_mask = N >= 10**6

    log_N = np.log(N[fit_mask])
    log_T = np.log(T[fit_mask])

    p, log_c = np.polyfit(log_N, log_T, 1)
    c = np.exp(log_c)

    # Smooth curve for the fitted model
    N_fit = np.logspace(
        np.log10(N.min()),
        np.log10(N.max()),
        200
    )
    T_fit = c * N_fit**p

    # =========================================================
    # Linear plot
    # =========================================================
    plt.figure(figsize=(10, 6))

    plt.plot(
        N,
        T,
        marker="o",
        label="Measured runtime"
    )

    plt.plot(
        N_fit,
        T_fit,
        linestyle="--",
        label=fr"Fit: $T \propto N^{{{p:.2f}}}$"
    )

    plt.xlabel("Number of photons (N)")
    plt.ylabel("Runtime (s)")
    plt.title("Runtime vs Number of Photons")

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.text(
        0.05,
        0.95,
        fr"Measured scaling: $T \propto N^{{{p:.3f}}}$",
        transform=plt.gca().transAxes,
        verticalalignment="top"
    )

    filepath = os.path.join(
        output_path,
        "runtime_linear.png"
    )

    plt.savefig(
        filepath,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    # =========================================================
    # Log-log plot
    # =========================================================
    plt.figure(figsize=(10, 6))

    plt.loglog(
        N,
        T,
        marker="o",
        label="Measured runtime"
    )

    plt.loglog(
        N_fit,
        T_fit,
        linestyle="--",
        label=fr"Power-law fit: $T \propto N^{{{p:.2f}}}$"
    )

    # Ideal O(N) reference line.
    # Anchor it at the last measured point.
    T_linear = T[-1] * (N_fit / N[-1])

    plt.loglog(
        N_fit,
        T_linear,
        linestyle=":",
        label=r"Ideal $O(N)$"
    )

    plt.xlabel("Number of photons (N)")
    plt.ylabel("Runtime (s)")
    plt.title("Runtime Scaling")

    plt.grid(True, which="both", alpha=0.3)
    plt.legend()

    plt.text(
        0.05,
        0.05,
        fr"Measured scaling exponent: $p = {p:.3f}$",
        transform=plt.gca().transAxes
    )

    filepath = os.path.join(
        output_path,
        "runtime_loglog.png"
    )

    plt.savefig(
        filepath,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Measured scaling exponent: p = {p:.4f}")