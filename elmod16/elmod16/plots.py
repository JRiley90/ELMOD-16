import matplotlib.pyplot as plt
import numpy as np

from .utils import make_constellation


def plot_constellation(a_k, phi_k, title="Constellation"):
    const = make_constellation(a_k, phi_k)
    plt.figure(figsize=(5, 5))
    plt.scatter(const.real, const.imag, marker="o")
    plt.axhline(0, color="grey", linewidth=0.5)
    plt.axvline(0, color="grey", linewidth=0.5)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_ber_curve(snr_list, ber_list, title="BER vs SNR"):
    plt.figure()
    plt.semilogy(snr_list, ber_list, marker="o")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title(title)
    plt.tight_layout()
    plt.show()
