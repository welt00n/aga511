import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURES_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
ONE = np.float32(1)


def naive_sine(cosine):
    return np.sqrt(np.maximum(ONE - cosine * cosine, 0))


def factored_sine(cosine):
    return np.sqrt(np.maximum((ONE - cosine) * (ONE + cosine), 0))


def exact_sine_of_stored_cosine(cosine):
    stored = cosine.astype(np.float64)
    return np.sqrt((1 - stored) * (1 + stored))


def relative_error(approximation, exact):
    return np.abs(approximation.astype(np.float64) - exact) / exact


def errors_by_angle(angles):
    cosines = np.cos(angles).astype(np.float32)
    exact = exact_sine_of_stored_cosine(cosines)
    usable = exact > 0
    return (angles[usable], relative_error(naive_sine(cosines), exact)[usable],
            relative_error(factored_sine(cosines), exact)[usable])


def worst_errors_near_one(floats_below_one=2 ** 16):
    float_spacing_below_one = 2.0 ** -24
    cosines = (1 - np.arange(1, floats_below_one + 1) * float_spacing_below_one).astype(np.float32)
    exact = exact_sine_of_stored_cosine(cosines)
    return relative_error(naive_sine(cosines), exact).max(), relative_error(factored_sine(cosines), exact).max()


def print_table(angles):
    print(f"{'angle (rad)':>12} {'cos (float32)':>14} {'naive rel. error':>17} {'factored rel. error':>20}")
    for angle in angles:
        cosine = np.array([np.cos(angle)], dtype=np.float32)
        exact = exact_sine_of_stored_cosine(cosine)
        print(f"{angle:12.0e} {cosine[0]:14.9f} {relative_error(naive_sine(cosine), exact)[0]:17.2e} "
              f"{relative_error(factored_sine(cosine), exact)[0]:20.2e}")


def plot_errors(path):
    angles, naive, factored = errors_by_angle(np.logspace(-4, np.log10(np.pi / 2), 20000))
    unit_roundoff = 2.0 ** -24
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.loglog(angles, naive, ".", markersize=1.5, alpha=0.5, label=r"naive: $\sqrt{1 - c^2}$")
    ax.loglog(angles, factored, ".", markersize=1.5, alpha=0.5, label=r"factored: $\sqrt{(1 - c)(1 + c)}$")
    ax.loglog(angles, unit_roundoff / (2 * angles ** 2), "k--", linewidth=1, label=r"envelope $2^{-25} / \theta^2$")
    ax.axvline(2.0 ** -12, color="red", linestyle=":",
               label=r"below $2^{-12} \approx 2.4\cdot10^{-4}$ rad, $\cos\theta$ rounds to 1")
    ax.axhline(unit_roundoff, color="gray", linestyle=":", label=r"FP32 unit roundoff $2^{-24}$")
    ax.set_xlabel(r"angle $\theta$ (rad), with $c = \cos\theta$ stored in FP32")
    ax.set_ylabel(r"relative error vs the exact sine of the stored $c$")
    ax.set_title("sine from cosine in FP32: the naive formula loses digits near the pole")
    ax.set_ylim(1e-9, 1)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(markerscale=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIGURES_FOLDER, exist_ok=True)
    print_table([1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.5, 1.0])
    naive_worst, factored_worst = worst_errors_near_one()
    print(f"\nworst over the 65536 floats just below 1: naive {naive_worst:.2e}, factored {factored_worst:.2e}")
    plot_errors(os.path.join(FIGURES_FOLDER, "sine_from_cosine_fp32.png"))
