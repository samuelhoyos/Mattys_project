import matplotlib.pyplot as plt
import numpy as np
from . import data


def plot_nmax_profiles(
    times_list: list,
    nmax_profiles_list: list,
    tfwhm_list: list,
    times_nmax_array: np.ndarray,
    nmax_array: np.ndarray,
):
    cmap = plt.cm.tab20
    n_curves = len(tfwhm_list)
    colors = [cmap(i) for i in range(n_curves)]
    plt.figure(figsize=(12, 8))
    for times, nmax_profile, tfwhm in zip(times_list, nmax_profiles_list, tfwhm_list):
        plt.plot(
            times * 1e15,
            nmax_profile,
            label=rf"$t_{{\text{{fwhm}}}}={tfwhm}$fs",
            color=colors[tfwhm_list.index(tfwhm)],
        )
    plt.plot(
        times_nmax_array * 1e15, nmax_array, linestyle="-", linewidth=3, color="black"
    )
    plt.xlabel("$t$ (fs)", fontsize=30)
    plt.ylabel("Max Positron Density", fontsize=30)
    plt.title("Max Positron Density Profiles Over Time", fontsize=30)
    plt.legend(loc="upper right", fontsize=25, frameon=False)
    plt.tight_layout()
    plt.show()


def plot_single_profile(times, nmax_profile, tfwhm):
    plt.figure(figsize=(12, 8))
    plt.plot(
        times * 1e15,
        nmax_profile,
        label=rf"$t_{{\text{{fwhm}}}}={tfwhm}$fs",
        color="blue",
    )
    plt.xlabel("$t$ (fs)", fontsize=30)
    plt.ylabel("Max Positron Density", fontsize=30)
    plt.title("Max Positron Density Profile Over Time", fontsize=30)
    plt.legend(loc="upper right", fontsize=25, frameon=False)
    plt.tight_layout()
    plt.show()
