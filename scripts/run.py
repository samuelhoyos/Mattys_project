import os
from pathlib import Path
from modules import data, plot
import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

tfwhm_list = [20, 30, 75]  # [
#     1,
#     5,
#     10,
#     15,
#     20,
#     25,
#     30,
#     50,
#     65,
#     70,
#     75
# ]  # , 70,75, 90,100, 120,150]#,1, 5, 6.5, 7.5, 10, 15, 20, 25, 30, 50, 65, 70, 75, 90, 100, 120, 150]


def create_file_name(tfwhm: float, binning: int = 128) -> str:
    return f"mattys_3D_massimo_moving_tfwhm{tfwhm}fs_binning{binning}"


if __name__ == "__main__":
    filenames_128 = [create_file_name(tfwhm) for tfwhm in tfwhm_list]
    densities_128 = [
        data.load_density("positrons", filename) for filename in filenames_128
    ]
    filenames_64 = [create_file_name(tfwhm, binning=64) for tfwhm in tfwhm_list]
    densities_64 = [
        data.load_density("positrons", filename) for filename in filenames_64
    ]
    # npon=data.load_density("positrons", filename)
    nponmax_profiles_128 = [data.get_nmax_profile(npon) for npon in densities_128]
    nponmax_profiles_64 = [data.get_nmax_profile(npon) for npon in densities_64]
    times_list_128 = [data.get_times(filename) for filename in filenames_128]
    times_list_64 = [data.get_times(filename) for filename in filenames_64]
    times_nmax_array, nmax_array = data.get_maxima_envelope(
        nponmax_profiles_128, times_list_128
    )
    times_nmax_array_128, nmax_array_128 = data.get_maxima_envelope(
        nponmax_profiles_128, times_list_128
    )
    times_nmax_array_64, nmax_array_64 = data.get_maxima_envelope(
        nponmax_profiles_64, times_list_64
    )
    # popt_long_128, popt_short_128 = data.fit_envelope(np.array(tfwhm_list), nmax_array_128)
    # popt_long_64, popt_short_64 = data.fit_envelope(np.array(tfwhm_list), nmax_array_64)
    minima_indices_128 = [
        data.find_n_local_minima(nponmax_profiles_128[i])
        for i in range(len(tfwhm_list))
    ]
    minima_indices_64 = [
        data.find_n_local_minima(nponmax_profiles_64[i]) for i in range(len(tfwhm_list))
    ]
    plt.figure(figsize=(12, 8))
    plot.plot_nmax_profiles(
        times_list_128,
        nponmax_profiles_128,
        tfwhm_list,
        times_nmax_array_128,
        nmax_array_128,
        minima_indices=minima_indices_128,
        nponmax_profiles=nponmax_profiles_128,
    )
    plot.plot_nmax_profiles(
        times_list_64,
        nponmax_profiles_64,
        tfwhm_list,
        times_nmax_array_64,
        nmax_array_64,
        legend=False,
        alpha=0.5,
    )
    times = [data.get_times(filename) for filename in filenames_128]
    N_pons = [data.get_N_pon(filename) for filename in filenames_128]
    average_sep = data.average_minima_separation(times_list_128, minima_indices_128)
    end_times = [data.get_end_of_collision(filename) for filename in filenames_128]
    collision_times = [data.get_collision_time(filename) for filename in filenames_128]

    print((np.array(end_times)) * 1e15)
    print(tfwhm_list)

    plot.times_end_of_collision(end_times, collision_times, tfwhm_list)
    plt.ylim(0, nmax_array_128.max())
    plt.xlim(0)
    plt.tight_layout()
    plt.show()
    # plt.figure(figsize=(12, 8))

    # plot.plot_nmax_tfwhm(
    #     tfwhm_list, nmax_array_128, popt_long=popt_long_128, popt_short=popt_short_128
    # )
    # plot.plot_nmax_tfwhm(
    #     tfwhm_list, nmax_array_64, popt_long=popt_long_64, popt_short=popt_short_64, alpha=0.3, binning=64
    # )
    # print(popt_long_128, popt_short_128)
    # print(popt_long_64, popt_short_64)
    input("Press Enter to exit...")
