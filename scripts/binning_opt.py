import os
from pathlib import Path
from modules import data, plot
import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

tfwhm_list = [1, 5, 10, 15, 20, 25, 30, 50, 65, 70, 75, 90, 100, 120, 150]


def create_file_name(tfwhm: float, binning: int = 128) -> str:
    return f"mattys_3D_massimo_moving_tfwhm{tfwhm}fs_tracking"


if __name__ == "__main__":
    filenames = [create_file_name(tfwhm) for tfwhm in tfwhm_list]
    binnings = np.arange(30, 200, 5)
    df_tracks = [
        data.get_tracks_df(filename=filename, species="positrons")
        for filename in filenames
    ]
    df_tracks = [
        data.filter_df_after_collision(filename=filename, species="positrons")
        for filename in filenames
    ]
    dfs_binned = [
        data.bin_df(df_track, filename=filename, binnings=binnings)
        for df_track, filename in zip(df_tracks, filenames)
    ]
    dfs_density = [
        data.get_density_dfs(df_binned, filename=filename, binnings=binnings)
        for df_binned, filename in zip(dfs_binned, filenames)
    ]
    dfs_max_n = [
        data.get_unified_maxdensity_df(df_density, binnings=binnings)
        for df_density in dfs_density
    ]

    # plt.plot(df_max_n["times"],df_max_n["n_55"])
    # plt.show()
    # input("Press Enter to exit...")
    cmap = plt.cm.tab20
    n_curves = len(tfwhm_list)
    colors = [cmap(i) for i in range(n_curves)]
    for i, df in enumerate(dfs_max_n):
        plt.plot(
            binnings,
            df.filter(regex="^n_").max(),
            color=colors[i],
            linestyle="-",
            marker="o",
            linewidth=3,
        )
    plt.show()
    input("Press Enter to exit...")
