import json
import os
import numpy as np
from datetime import datetime

from .config import M_ORDER

DEFAULT_REGISTRY_PATH = "elmod16_registry.json"


def _to_list(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def save_entry(
    genome,
    metrics,
    version="v1",
    snr_db=12,
    family="unknown",
    path=DEFAULT_REGISTRY_PATH,
    author="Jake Harry Riley",
):
    """
    genome: GenomeV1 or GenomeV2
    metrics: dict with keys like 'score','ber','eta','papr'
    """
    entry = {
        "id": f"EL16-{version}-SNR{snr_db}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
        "author": author,
        "date": datetime.utcnow().strftime("%Y-%m-%d"),
        "version": version,
        "snr_db": snr_db,
        "family": family,
        "M": M_ORDER,
        "a_k": _to_list(genome.a_k),
        "phi_k": _to_list(genome.phi_k),
        "rolloff": float(genome.rolloff),
        "packing": float(genome.packing),
        "metrics": {k: float(v) for k, v in metrics.items()},
    }

    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(entry)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    return entry
