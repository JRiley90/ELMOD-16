import numpy as np
from dataclasses import dataclass
from .utils import make_constellation, rrc_pulse, add_awgn, pam_papr, spectral_efficiency
from .config import M_ORDER, SPS, SPAN, GENERATION_SYMBOLS, SNR_DB_DEFAULT

# ----------------------------------------------------------
# V1 GENOME STRUCTURE
# ----------------------------------------------------------

@dataclass
class GenomeV1:
    a_k: np.ndarray       # Amplitude genes
    phi_k: np.ndarray     # Phase genes
    rolloff: float        # Pulse rolloff
    packing: float        # Nyquist packing

    def clone(self):
        return GenomeV1(
            a_k=self.a_k.copy(),
            phi_k=self.phi_k.copy(),
            rolloff=float(self.rolloff),
            packing=float(self.packing)
        )


def random_genome_v1():
    return GenomeV1(
        a_k=np.random.uniform(0.1, 2.0, M_ORDER),
        phi_k=np.random.uniform(0, 2*np.pi, M_ORDER),
        rolloff=np.random.uniform(0.05, 0.95),
        packing=np.random.uniform(0.9, 1.1)
    )


# ----------------------------------------------------------
# FITNESS EVALUATION
# ----------------------------------------------------------

def evaluate_v1(genome, snr_db=SNR_DB_DEFAULT):
    """
    Full AWGN evaluation with pulse shaping + matched filter +
    symbol-by-symbol detection.
    Returns:
        (score, ber, eta, papr)
    """

    # 1. Build constellation
    const = make_constellation(genome.a_k, genome.phi_k)

    # 2. Random symbol generation
    sym_idx = np.random.randint(0, M_ORDER, GENERATION_SYMBOLS)
    symbols = const[sym_idx]

    # 3. Pulse shaping
    pulse = rrc_pulse(genome.rolloff, sps=SPS, span=SPAN)

    # Upsample
    up = np.zeros(len(symbols) * SPS, dtype=complex)
    up[::SPS] = symbols

    # Convolution (Tx)
    tx = np.convolve(up, pulse, mode='same')

    # 4. Channel (AWGN)
    rx = add_awgn(tx, snr_db)

    # 5. Matched filter
    mf = np.convolve(rx, pulse, mode='same')

    # 6. Downsample
    rxs = mf[::SPS][10:-10]  # drop edges
    tx_ref = sym_idx[10:-10]

    # 7. Detection
    distances = np.abs(rxs[:, None] - const[None, :])
    dec = np.argmin(distances, axis=1)

    # 8. Metrics
    ser = np.mean(dec != tx_ref)
    ber = max(ser / np.log2(M_ORDER), 1e-6)
    eta = spectral_efficiency(M_ORDER, genome.packing)
    papr = pam_papr(tx)

    # 9. Fitness
    score = (3 * eta) + abs(-np.log10(ber)) - (0.1 * papr)

    return score, ber, eta, papr


# ----------------------------------------------------------
# MUTATION + CROSSOVER
# ----------------------------------------------------------

def mutate_v1(g, rate=0.05):
    g = g.clone()

    g.a_k += np.random.normal(0, rate, M_ORDER)
    g.phi_k += np.random.normal(0, rate, M_ORDER)
    g.rolloff += np.random.normal(0, rate)
    g.packing += np.random.normal(0, rate / 2)

    # Clamp/wrap
    g.a_k = np.clip(g.a_k, 0.1, 4.0)
    g.phi_k = np.mod(g.phi_k, 2*np.pi)
    g.rolloff = np.clip(g.rolloff, 0.05, 0.95)
    g.packing = np.clip(g.packing, 0.9, 1.1)

    return g


def crossover_v1(a, b):
    c = a.clone()

    mask = np.random.rand(M_ORDER) > 0.5
    c.a_k[mask] = b.a_k[mask]
    c.phi_k[mask] = b.phi_k[mask]

    if np.random.rand() > 0.5:
        c.rolloff = b.rolloff
    if np.random.rand() > 0.5:
        c.packing = b.packing

    return c


# ----------------------------------------------------------
# FULL EVOLUTION LOOP (V1)
# ----------------------------------------------------------

def evolve_v1(pop_size=50, generations=60, verbose=True):
    pop = [random_genome_v1() for _ in range(pop_size)]
    best = None
    best_score = -999

    for g in range(generations):
        results = []
        for genome in pop:
            score, ber, eta, papr = evaluate_v1(genome)
            results.append((genome, score, ber, eta, papr))

        results.sort(key=lambda x: x[1], reverse=True)
        elite = results[:5]

        if elite[0][1] > best_score:
            best_score = elite[0][1]
            best = elite[0][0].clone()

        if verbose:
            print(
                f"Gen {g+1:02d} | "
                f"Score={elite[0][1]:.2f} | "
                f"BER={elite[0][2]:.6f} | "
                f"η={elite[0][3]:.2f} | "
                f"PAPR={elite[0][4]:.2f}"
            )

        # Breeding
        survivors = [e[0] for e in elite]
        children = []

        while len(children) < pop_size - 5 - 5:
            p1 = survivors[np.random.randint(0, len(survivors))]
            p2 = survivors[np.random.randint(0, len(survivors))]
            child = mutate_v1(crossover_v1(p1, p2))
            children.append(child)

        randoms = [random_genome_v1() for _ in range(5)]
        pop = survivors + children + randoms

    return best, best_score
