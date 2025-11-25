import numpy as np
from dataclasses import dataclass

from .utils import (
    make_constellation,
    rrc_pulse,
    add_awgn,
    pam_papr,
    spectral_efficiency,
)
from .config import (
    M_ORDER,
    SPS,
    SPAN,
    GENERATION_SYMBOLS,
    SNR_DB_DEFAULT,
)

# ----------------------------------------------------------
# V2 GENOME (same genes as V1, but used in FTN/ISI context)
# ----------------------------------------------------------

@dataclass
class GenomeV2:
    a_k: np.ndarray
    phi_k: np.ndarray
    rolloff: float
    packing: float  # < 1.0 => FTN

    def clone(self):
        return GenomeV2(
            a_k=self.a_k.copy(),
            phi_k=self.phi_k.copy(),
            rolloff=float(self.rolloff),
            packing=float(self.packing),
        )


def random_genome_v2():
    return GenomeV2(
        a_k=np.random.uniform(0.1, 2.0, M_ORDER),
        phi_k=np.random.uniform(0, 2*np.pi, M_ORDER),
        rolloff=np.random.uniform(0.05, 0.95),
        packing=np.random.uniform(0.7, 1.0),  # encourage FTN
    )

# ----------------------------------------------------------
# ISI CHANNEL MODEL
# ----------------------------------------------------------

def _effective_channel_taps(pulse, step, L=3):
    """
    Builds an approximate symbol-spaced channel from the
    matched-filtered pulse response.
    """
    g = np.convolve(pulse, pulse)  # matched filter equivalent
    center = len(g) // 2
    taps = []

    for k in range(L):
        idx = center + (k - (L//2)) * step
        if 0 <= idx < len(g):
            taps.append(g[idx])
        else:
            taps.append(0.0)

    taps = np.array(taps, dtype=complex)
    # normalize so main tap ~ 1
    if np.abs(taps[L//2]) > 0:
        taps /= taps[L//2]
    return taps


# ----------------------------------------------------------
# MLSE-LITE (VITERBI) DETECTOR
# ----------------------------------------------------------

def viterbi_mlse(y, const, taps, snr_db):
    """
    Simple MLSE Viterbi detector with memory L-1.
    Assumes:
        y[k] = sum_{l=0..L-1} taps[l] * x[k-l] + n[k]
    where x[k] are constellation symbols from 'const'.
    """
    L = len(taps)
    M = len(const)
    mem = L - 1
    N = len(y)

    if mem <= 0:
        # fall back to symbol-by-symbol
        distances = np.abs(y[:, None] - const[None, :])
        return np.argmin(distances, axis=1)

    n_states = M**mem

    # Precompute state -> symbol history (indices)
    def state_to_hist(s):
        hist = []
        for _ in range(mem):
            hist.append(s % M)
            s //= M
        return hist  # [x_{k-1}, x_{k-2}, ...]

    # noise variance approx (constellation avg power ~ 1)
    snr_linear = 10**(snr_db / 10)
    noise_var = 1.0 / snr_linear

    # Metrics
    INF = 1e18
    path_metric = np.ones(n_states) * INF
    path_metric[0] = 0.0  # start in all-zero state
    prev_state = np.zeros((N, n_states), dtype=int)
    prev_symbol = np.zeros((N, n_states), dtype=int)

    for k in range(N):
        new_metric = np.ones(n_states) * INF
        new_prev_state = np.zeros(n_states, dtype=int)
        new_prev_symbol = np.zeros(n_states, dtype=int)

        for s in range(n_states):
            if path_metric[s] >= INF / 2:
                continue  # unreachable

            hist = state_to_hist(s)  # length mem

            for m_idx in range(M):
                # new symbol index
                x0 = m_idx
                # new state encodes [x0, hist[0], hist[1], ...]
                new_s = 0
                mult = 1
                all_syms = [x0] + hist
                for idx in all_syms[:mem]:
                    new_s += idx * mult
                    mult *= M

                # predicted output sample
                x_syms = [const[x0]] + [const[h] for h in hist]
                x_syms = np.array(x_syms, dtype=complex)
                y_hat = np.sum(taps * x_syms[:L])

                # branch metric
                bm = np.abs(y[k] - y_hat)**2 / noise_var
                cand = path_metric[s] + bm

                if cand < new_metric[new_s]:
                    new_metric[new_s] = cand
                    new_prev_state[new_s] = s
                    new_prev_symbol[new_s] = m_idx

        path_metric = new_metric
        prev_state[k, :] = new_prev_state
        prev_symbol[k, :] = new_prev_symbol

    # Traceback
    end_state = np.argmin(path_metric)
    decoded = []
    s = end_state
    for k in range(N-1, -1, -1):
        sym = prev_symbol[k, s]
        decoded.append(sym)
        s = prev_state[k, s]
    decoded.reverse()
    return np.array(decoded, dtype=int)


# ----------------------------------------------------------
# V2 FITNESS EVALUATION
# ----------------------------------------------------------

def evaluate_v2(genome, snr_db=SNR_DB_DEFAULT, L_taps=3):
    """
    FTN/ISI channel + MLSE-lite detection.
    Returns: (score, ber, eta, papr)
    """
    const = make_constellation(genome.a_k, genome.phi_k)

    # Data
    sym_idx = np.random.randint(0, M_ORDER, GENERATION_SYMBOLS)
    symbols = const[sym_idx]

    # Pulse
    pulse = rrc_pulse(genome.rolloff, sps=SPS, span=SPAN)

    # FTN spacing (symbol spacing in samples)
    step = max(1, int(round(genome.packing * SPS)))
    N = len(symbols)
    length = N * step + len(pulse) + 4 * step
    tx = np.zeros(length, dtype=complex)

    # Place symbols with step samples apart
    start = 2 * step
    for i, s in enumerate(symbols):
        tx[start + i*step] += s

    # Shape
    tx = np.convolve(tx, pulse, mode="same")

    # Channel
    rx = add_awgn(tx, snr_db)

    # Matched filter
    mf = np.convolve(rx, pulse, mode="same")

    # Sampled sequence
    center = len(mf) // 2
    start_idx = start + (len(pulse)//2)
    y = []
    for i in range(N):
        idx = start_idx + i*step
        if 0 <= idx < len(mf):
            y.append(mf[idx])
    y = np.array(y, dtype=complex)

    # Compute effective taps for MLSE
    taps = _effective_channel_taps(pulse, step, L=L_taps)

    # Decode with Viterbi MLSE-lite
    dec_idx = viterbi_mlse(y, const, taps, snr_db)

    # Cut to common length
    Lmin = min(len(dec_idx), len(sym_idx))
    dec_idx = dec_idx[:Lmin]
    ref_idx = sym_idx[:Lmin]

    ser = np.mean(dec_idx != ref_idx)
    ber = max(ser / np.log2(M_ORDER), 1e-6)

    eta = spectral_efficiency(M_ORDER, genome.packing)
    papr = pam_papr(tx)

    score = (3 * eta) + abs(-np.log10(ber)) - (0.15 * papr)

    return score, ber, eta, papr


# ----------------------------------------------------------
# EVOLUTION LOOP (V2)
# ----------------------------------------------------------

def mutate_v2(g, rate=0.05):
    g = g.clone()

    g.a_k += np.random.normal(0, rate, M_ORDER)
    g.phi_k += np.random.normal(0, rate, M_ORDER)
    g.rolloff += np.random.normal(0, rate)
    g.packing += np.random.normal(0, rate / 3)

    g.a_k = np.clip(g.a_k, 0.1, 4.0)
    g.phi_k = np.mod(g.phi_k, 2*np.pi)
    g.rolloff = np.clip(g.rolloff, 0.05, 0.95)
    g.packing = np.clip(g.packing, 0.7, 1.0)

    return g


def crossover_v2(a, b):
    c = a.clone()
    mask = np.random.rand(M_ORDER) > 0.5
    c.a_k[mask] = b.a_k[mask]
    c.phi_k[mask] = b.phi_k[mask]

    if np.random.rand() > 0.5:
        c.rolloff = b.rolloff
    if np.random.rand() > 0.5:
        c.packing = b.packing
    return c


def evolve_v2(pop_size=40, generations=40, verbose=True):
    pop = [random_genome_v2() for _ in range(pop_size)]
    best = None
    best_score = -999

    for g in range(generations):
        results = []
        for genome in pop:
            score, ber, eta, papr = evaluate_v2(genome)
            results.append((genome, score, ber, eta, papr))

        results.sort(key=lambda x: x[1], reverse=True)
        elite = results[:5]

        if elite[0][1] > best_score:
            best_score = elite[0][1]
            best = elite[0][0].clone()

        if verbose:
            print(
                f"[V2] Gen {g+1:02d} | "
                f"Score={elite[0][1]:.2f} | "
                f"BER={elite[0][2]:.6f} | "
                f"η={elite[0][3]:.2f} | "
                f"PAPR={elite[0][4]:.2f}"
            )

        survivors = [e[0] for e in elite]
        children = []

        while len(children) < pop_size - 5 - 5:
            p1 = survivors[np.random.randint(0, len(survivors))]
            p2 = survivors[np.random.randint(0, len(survivors))]
            child = mutate_v2(crossover_v2(p1, p2))
            children.append(child)

        randoms = [random_genome_v2() for _ in range(5)]
        pop = survivors + children + randoms

    return best, best_score
