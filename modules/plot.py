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
    # plt.plot(
    #     times_nmax_array * 1e15, nmax_array, linestyle="-", linewidth=3, color="black"
    # )
   
    plt.xlabel("$t$ (fs)", fontsize=30)
    plt.ylabel("Max Positron Density", fontsize=30)
    plt.title("Max Positron Density Profiles Over Time", fontsize=30)
    plt.legend(loc="best", fontsize=25, frameon=False)
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


def plot_nmax_tfwhm(
    tfwhm_list: list,
    nmax_array: np.ndarray,
    popt_long: np.ndarray = None,
    popt_short: np.ndarray = None,
):

    mask = list(nmax_array).index(np.max(nmax_array))
    tfwhm_list_short = np.array(tfwhm_list)[:mask]
    tfwhm_list_long = np.array(tfwhm_list)[mask:]
    print(tfwhm_list_long)
    plt.figure(figsize=(12, 8))
    plt.plot(
        list(tfwhm_list),
        nmax_array,
        marker="o",
        linestyle="",
        color="blue",
        label="Data",
    )
    if popt_long is not None:

        plt.plot(
            tfwhm_list_long,
            data.long_time(tfwhm_list_long, *popt_long),
            linestyle="--",
            linewidth=3,
            color="red",
            # label=f'${popt_long[0]}/(t_{{fwhm}}-{popt_long[1]})+{{{popt_long[2]}}}$'
        )
    if popt_short is not None:
        plt.plot(
            tfwhm_list_short,
            data.short_time(tfwhm_list_short, *popt_short),
            linestyle="--",
            linewidth=3,
            color="red",
            # label=f'${popt_short[0]:.2f}+{popt_short[1]:.2f}t_{{fwhm}}$'
        )

    plt.xlabel("$t_{fwhm}$ (fs)", fontsize=30)
    plt.ylabel("Max Positron Density", fontsize=30)
    plt.title("Max Positron Density vs Pulse Duration", fontsize=30)
    plt.tick_params(axis="both", which="major", labelsize=25)
    plt.tight_layout()
    plt.legend(loc="upper right", fontsize=25, frameon=False)
    plt.show()
