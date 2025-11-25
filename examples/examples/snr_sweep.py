import numpy as np

from elmod16.v1_engine import evolve_v1
from elmod16.analysis import snr_sweep_v1
from elmod16.plots import plot_ber_curve

if __name__ == "__main__":
    # evolve a good V1 constellation first
    best, best_score = evolve_v1(pop_size=30, generations=20, verbose=True)
    print("Best score:", best_score)

    # Sweep SNR
    snr_list, ber_list, _ = snr_sweep_v1(best)
    print("SNR:", snr_list)
    print("BER:", ber_list)

    # Plot
    plot_ber_curve(snr_list, ber_list, title="ELMOD-16 V1: BER vs SNR")
