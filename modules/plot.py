import matplotlib.pyplot as plt
import numpy as np
from . import data


def plot_nmax_profiles(
    times_list: list,
    nmax_profiles_list: list,
    tfwhm_list: list,
    times_nmax_array: np.ndarray,
    nmax_array: np.ndarray,
    linestyle: str = "-",
    linewidth: int = 3,
    alpha: float = 1.0,
    legend: bool = True,
    minima_indices: list = None,
    nponmax_profiles: list = None,
):
    cmap = plt.cm.tab20
    n_curves = len(tfwhm_list)
    colors = [cmap(i) for i in range(n_curves)]
    # plt.figure(figsize=(12, 8))
    for times, nmax_profile, tfwhm in zip(times_list, nmax_profiles_list, tfwhm_list):
        plt.plot(
            times * 1e15,
            nmax_profile,
            label=rf"$t_{{\text{{fwhm}}}}={tfwhm}$fs",
            color=colors[tfwhm_list.index(tfwhm)],
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
        )
        if minima_indices is not None and nponmax_profiles is not None:
            idx = tfwhm_list.index(tfwhm)
            plt.plot(
                times[minima_indices[idx]] * 1e15,
                nponmax_profiles[idx][minima_indices[idx]],
                "o",
                color=colors[idx],
                alpha=alpha,
            )

    # plt.plot(
    #     times_nmax_array * 1e15, nmax_array, linestyle="-", linewidth=3, color="black"
    # )

    plt.xlabel("$t$ (fs)", fontsize=30)
    plt.ylabel("Max Positron Density", fontsize=30)
    plt.title("Max Positron Density Profiles Over Time", fontsize=30)
    if legend:
        plt.legend(loc="best", fontsize=25, frameon=False)
    plt.tight_layout()


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
    linestyle: str = "-.",
    linewidth: int = 3,
    marker: str = "o",
    alpha: float = 1.0,
    binning: int = 128,
):

    mask = list(nmax_array).index(np.max(nmax_array))
    tfwhm_list_short = np.array(tfwhm_list)[:mask]
    tfwhm_list_long = np.array(tfwhm_list)[mask:]
    print(tfwhm_list_long)

    plt.plot(
        list(tfwhm_list),
        nmax_array,
        marker=marker,
        linestyle="-",
        color="blue",
        label=f"Data for binning {binning}",
        alpha=alpha,
    )
    # if popt_long is not None:

    #     plt.plot(
    #         tfwhm_list_long,
    #         data.long_time(tfwhm_list_long, *popt_long),
    #         linestyle=linestyle,
    #         linewidth=linewidth,
    #         color="red",
    #         alpha=alpha,
    #         # label=f'${popt_long[0]}/(t_{{fwhm}}-{popt_long[1]})+{{{popt_long[2]}}}$'
    #     )
    # if popt_short is not None:
    #     plt.plot(
    #         tfwhm_list_short,
    #         data.short_time(tfwhm_list_short, *popt_short),
    #         linestyle=linestyle,
    #         linewidth=linewidth,
    #         color="red",
    #         alpha=alpha,
    #         # label=f'${popt_short[0]:.2f}+{popt_short[1]:.2f}t_{{fwhm}}$'
    #     )

    plt.xlabel("$t_{fwhm}$ (fs)", fontsize=30)
    plt.ylabel("Max Positron Density", fontsize=30)
    plt.title("Max Positron Density vs Pulse Duration", fontsize=30)
    plt.tick_params(axis="both", which="major", labelsize=25)
    plt.tight_layout()
    plt.legend(loc="upper right", fontsize=25, frameon=False)


def N_pon(N_pon: np.array, times: np.array, color: str):

    plt.plot(times * 1e15, N_pon, color=color)


def times_end_of_collision(
    end_of_collision_times: list,
    tfwhm_list: list,
    linestyle: str = "--",
    linewidth: int = 3,
    alpha: float = 1,
):

    cmap = plt.cm.tab20
    n_curves = len(tfwhm_list)
    colors = [cmap(i) for i in range(n_curves)]
    # plt.figure(figsize=(12, 8))
    for i, end_time in enumerate(end_of_collision_times):
        plt.vlines(
            x=(end_time) * 1e15,
            ymin=0,
            ymax=1,
            color=colors[tfwhm_list.index(tfwhm_list[i])],
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
        )
