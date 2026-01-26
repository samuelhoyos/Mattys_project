import happi
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import os

data_path = Path(__file__).parent.parent / "data"


def load_density(species: str, filename: str) -> np.array:
    file_path = data_path / filename
    S = happi.Open(f"{file_path}", verbose=False)
    species_dict = {"electrons": 1, "positrons": 0, "seed": 2}
    n = S.ParticleBinning(diagNumber=species_dict[species])

    return np.array(n.getData())

def get_collision_time(filename:str):
    file_path = data_path / filename
    os.environ["TFWHM_FS"] = filename.split("tfwhm")[1].split("f")[0]
    S = happi.Open(f"{file_path}", verbose=False)
    return (((S.namelist.v0)/(S.namelist.c_normalized+S.namelist.v0))*S.namelist.tfwhm + S.namelist.x_focus_laser/S.namelist.c_normalized)/S.namelist.omega0
    
def get_times(filename: str) -> np.array:
    collision_time=get_collision_time(filename)
    file_path = data_path / filename
    S = happi.Open(f"{file_path}", verbose=False)
    times = S.Scalar("Ntot_pon").getTimes()
    print(S.namelist.tfwhm_fs)
    # z=np.array(S.Scalar("Ntot_pon").getData())
    # indx=np.where(z>0)[0][0]+

    return (times / S.namelist.omega0)-collision_time


def get_nmax_profile(n: np.ndarray) -> np.ndarray:
    nmax_profile = np.max(n, axis=(1, 2))
    return nmax_profile


def get_maxima_envelope(
    nmax_profiles_list: list, times_list: list
) -> tuple[np.ndarray, np.ndarray]:
    nmax_list = [np.max(nmax_profile) for nmax_profile in nmax_profiles_list]
    times_nmax_list = [
        times[np.argmax(nmax_profile)]
        for times, nmax_profile in zip(times_list, nmax_profiles_list)
    ]
    return np.array(times_nmax_list), np.array(nmax_list)


def get_field(
    component: str,
    filename: str,
    subset_x: list = None,
    subset_y: list = None,
    subset_z: list = None,
) -> np.array:
    file_path = data_path / filename
    S = happi.Open(f"{file_path}", verbose=False)
    if subset_x is None and subset_y is None and subset_z is None:
        f = S.Field(0, f"{component}")
    else:
        f = S.Field(
            0, f"{component}", subset={"x": subset_x, "y": subset_y, "z": subset_z}
        )
    return np.array(f.getData())


def long_time(tfwhm, A, B, C):
    return (A / ((tfwhm + B))) - C


def short_time(tfwhm, A, B):
    return A + tfwhm * B


def fit_envelope(
    tfwhm_list: list,
    nmax_array: np.ndarray,
):
    mask = list(nmax_array).index(np.max(nmax_array))
    tfwhm_list_short = np.array(tfwhm_list)[:mask]
    tfwhm_list_long = np.array(tfwhm_list)[mask:]

    init_long = [1, 0, 1]
    init_short = [0, 1]
    popt_long, pcov_long = curve_fit(
        long_time, tfwhm_list_long, nmax_array[mask:], p0=init_long, maxfev=800000
    )
    popt_short, pcov_short = curve_fit(
        short_time, tfwhm_list_short, nmax_array[:mask], p0=init_short, maxfev=800000
    )
    return popt_long, popt_short


def check_density_planes(
    n: np.ndarray, plane: str = "xy", index: int = 0
) -> pd.DataFrame:
    if plane == "xy":
        data_plane = n[:, :, :, index]
    elif plane == "xz":
        data_plane = n[:, :, index, :]
    elif plane == "yz":
        data_plane = n[:, index, :, :]
    else:
        raise ValueError("Invalid plane specified. Choose from 'xy', 'xz', or 'yz'")

    return data_plane


def find_n_local_minima(n_max_profile: np.ndarray, prominence: float = 0.05):

    scale = np.max(n_max_profile) - np.min(n_max_profile)

    inverted_profile = -n_max_profile
    peaks, properties = find_peaks(inverted_profile, prominence=prominence * scale)

    return peaks


def average_minima_separation(
    time_list: list,
    minima_indices: list,
) -> float:
    separations = []
    for times, minima_index in zip(
        time_list, minima_indices
    ):  
        print(minima_index)
        if len(minima_index) >= 2:
            separation = np.mean(np.diff(times[minima_index]))
            separations.append(separation)

    if separations:
        return np.array(separations)*1e15
    else:
        raise ValueError("No minima found to calculate separation.")
    
def get_N_pon(filename:str):

    file_path = data_path / filename
    S = happi.Open(f"{file_path}", verbose=False)
    species_dict = {"electrons": 1, "positrons": 0, "seed": 2}
    Npon = S.Scalar("Ntot_pon")
 
    return np.array(Npon.getData())
    
    
