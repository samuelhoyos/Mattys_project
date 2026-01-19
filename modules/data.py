import happi
import numpy as np
import pandas as pd
from pathlib import Path

data_path = Path(__file__).parent.parent / "data"


def load_density(species: str, filename: str) -> np.array:
    file_path = data_path / filename
    S = happi.Open(f"{file_path}", verbose=False)
    species_dict = {"electrons": 1, "positrons": 0, "seed": 2}
    n = S.ParticleBinning(diagNumber=species_dict[species])
    return np.array(n.getData())


def get_times(filename: str) -> np.array:
    file_path = data_path / filename
    S = happi.Open(f"{file_path}", verbose=False)
    times = S.Scalar("Ntot_pon").getTimes()
    return times / S.namelist.omega0


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

def get_field(component: str, filename: str, subset_x:list=None,subset_y:list=None, subset_z:list=None) -> np.array:
    file_path = data_path / filename
    S = happi.Open(f"{file_path}", verbose=False)
    if  subset_x is None and subset_y is None and subset_z is None:
            f = S.Field(0, f"{component}")
    else:
        f = S.Field(0, f"{component}", subset={
            'x': subset_x,
            'y': subset_y,
            'z': subset_z
        })
    return np.array(f.getData())

def long_time(tfwhm,A,B,C):
        return (A/((tfwhm+B))) - C
    
def short_time(tfwhm,A,B):
        return A+tfwhm*B

def fit_envelope(tfwhm_list:list,nmax_array: np.ndarray,):
    mask=list(nmax_array).index(np.max(nmax_array))
    tfwhm_list_short= np.array(tfwhm_list)[:mask]
    tfwhm_list_long= np.array(tfwhm_list)[mask:]
    
    init_long=[1,0,1]
    init_short=[0,1]
    popt_long, pcov_long = curve_fit(long_time, tfwhm_list_long, nmax_array[mask:], p0=init_long,  maxfev = 800000)
    popt_short, pcov_short = curve_fit(short_time, tfwhm_list_short, nmax_array[:mask], p0=init_short, maxfev = 800000)
    return popt_long, popt_short

