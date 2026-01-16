import matplotlib.pyplot as plt



def plot_nmax_profiles(times_list, nmax_profiles_list, tfwhm_list):
    plt.figure(figsize=(10, 6))
    for times, nmax_profile, tfwhm in zip(times_list, nmax_profiles_list, tfwhm_list):
        plt.plot(times, nmax_profile, label=f"tfwhm={tfwhm} fs")
    plt.xlabel("Time (1/omega0)")
    plt.ylabel("Max Positron Density")
    plt.title("Max Positron Density Profiles Over Time")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()