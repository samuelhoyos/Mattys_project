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
