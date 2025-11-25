import numpy as np

# -----------------------------
# Complex Math Utilities
# -----------------------------

def make_constellation(a_k, phi_k):
    """
    Converts amplitude + phase vectors into normalized complex constellation points.
    """
    pts = a_k * np.exp(1j * phi_k)
    avg_power = np.mean(np.abs(pts)**2)
    return pts / np.sqrt(avg_power)


# -----------------------------
# Pulse Generation (RRC)
# -----------------------------

def rrc_pulse(beta, sps=8, span=6):
    """
    Root Raised Cosine (RRC) pulse with fully normalized energy.
    Handles singularities at t=0 and t = ±T/(4β).
    """
    t = np.arange(-span * sps, span * sps + 1) / sps

    with np.errstate(divide='ignore', invalid='ignore'):
        num = np.sin(np.pi * t * (1 - beta)) + \
              4 * beta * t * np.cos(np.pi * t * (1 + beta))
        den = np.pi * t * (1 - (4 * beta * t)**2)
        pulse = num / den

    # Fix t = 0
    pulse[t == 0] = 1 - beta + (4 * beta / np.pi)

    # Fix t = ±1/(4β)
    if beta > 0:
        singular_idx = np.isclose(np.abs(t), 1/(4*beta))
        pulse[singular_idx] = (beta / np.sqrt(2)) * (
            (1 + 2/np.pi) * np.sin(np.pi/(4*beta)) +
            (1 - 2/np.pi) * np.cos(np.pi/(4*beta))
        )

    # Normalize energy
    pulse = pulse / np.sqrt(np.sum(pulse**2))
    return pulse


# -----------------------------
# AWGN Noise
# -----------------------------

def add_awgn(signal, snr_db):
    """
    Adds complex AWGN noise to a signal.
    """
    snr_linear = 10**(snr_db / 10)
    power = np.mean(np.abs(signal)**2)
    noise_power = power / snr_linear
    sigma = np.sqrt(noise_power / 2)
    noise = sigma * (np.random.randn(len(signal)) + 1j*np.random.randn(len(signal)))
    return signal + noise


# -----------------------------
# Metric Helpers
# -----------------------------

def pam_papr(x):
    """
    PAPR of a complex vector.
    """
    return np.max(np.abs(x)**2) / np.mean(np.abs(x)**2)


def spectral_efficiency(M, packing):
    return np.log2(M) / packing
