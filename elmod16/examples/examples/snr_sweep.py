import numpy as np
from elmod16 import evolutionary_engine_v1

SNR_RANGE = np.linspace(-5, 25, 16)

def run_snr_sweep():
    results = []
    for snr_db in SNR_RANGE:
        ber = evolutionary_engine_v1.evaluate_constellation(
            snr_db=snr_db
        )
        results.append((snr_db, ber))
        print(f"SNR {snr_db:.1f} dB -> BER {ber:.3e}")

    return results

if __name__ == "__main__":
    run_snr_sweep()
