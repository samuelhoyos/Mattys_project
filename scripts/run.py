import os
from pathlib import Path
from modules import data, plot
import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

tfwhm_list = [1, 5, 6.5, 7.5, 10, 15, 20, 25, 30, 50, 65, 70, 75, 90, 100, 120, 150]


def create_file_name(tfwhm: float) -> str:
    return f"mattys_3D_massimo_moving_tfwhm{tfwhm}fs"


if __name__ == "__main__":
    filenames = [create_file_name(tfwhm) for tfwhm in tfwhm_list]
    densities = [data.load_density("positrons", filename) for filename in filenames]
    # npon=data.load_density("positrons", filename)
    nponmax_profiles = [data.get_nmax_profile(npon) for npon in densities]
    times_list = [data.get_times(filename) for filename in filenames]
    times_nmax_array, nmax_array = data.get_maxima_envelope(
        nponmax_profiles, times_list
    )
    plot.plot_nmax_profiles(
        times_list, nponmax_profiles, tfwhm_list, times_nmax_array, nmax_array
    )

    input("Press Enter to exit...")
