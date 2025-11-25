import numpy as np

from .v1_engine import evaluate_v1
from .v2_engine import evaluate_v2
from .config import SNR_DB_DEFAULT

def snr_sweep_v1(genome, snr_list=None):
    if snr_list is None:
        snr_list = list(range(6, 21, 2))
    ber_list = []
    score_list = []
    for snr in snr_list:
        score, ber, eta, papr = evaluate_v1(genome, snr_db=snr)
        ber_list.append(ber)
        score_list.append(score)
    return np.array(snr_list), np.array(ber_list), np.array(score_list)


def snr_sweep_v2(genome, snr_list=None):
    if snr_list is None:
        snr_list = list(range(6, 21, 2))
    ber_list = []
    score_list = []
    for snr in snr_list:
        score, ber, eta, papr = evaluate_v2(genome, snr_db=snr)
        ber_list.append(ber)
        score_list.append(score)
    return np.array(snr_list), np.array(ber_list), np.array(score_list)
