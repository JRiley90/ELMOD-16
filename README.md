# ELMOD-16 v1.0
## Evolutionary Light Modulation Discovery Framework (16-Point)

**Author:** Jake Harry Riley  
**Independent Researcher (UK)**  
**Created:** 25 November 2025

---

## Overview

ELMOD-16 is a complete evolutionary research framework for discovering novel 16-point digital modulation formats using AI-assisted optimization.  
It includes:

- **V1:** AWGN evolutionary engine  
- **V2:** FTN/ISI + MLSE-lite evolutionary engine  
- **Novelty scoring + analysis tools**  
- **Constellation classifier**  
- **Experiment harness (SNR sweep)**  
- **Registry + plotting tools**  

The framework explores both orthogonal and non-orthogonal signaling regimes and has been designed for research reproducibility and extension.

---

## Key Features

### **V1 – AWGN Engine**
- Evolutionary search over amplitude, phase, roll-off, and packing  
- RRC pulse shaping  
- Symbol-by-symbol detection  
- Scoring: spectral efficiency, BER, PAPR  

### **V2 – FTN/ISI + MLSE Engine**
- Sub-Nyquist packing (p < 1.0)  
- Controlled ISI  
- Tap estimation  
- MLSE-lite Viterbi sequence detector  
- Complexity-aware fitness  

---

## Repository Structure

ELMOD-16/ │ ├── README.md ├── LICENSE ├── requirements.txt │ ├── docs/ │   └── ELMOD-16_v1.0_Spec.pdf │ ├── elmod16/ │   ├── v1_engine.py │   ├── v2_engine.py │   ├── analysis.py │   ├── classifier.py │   ├── registry.py │   ├── plots.py │   ├── utils.py │   └── config.py │ └── examples/ ├── run_v1_demo.py ├── run_v2_demo.py ├── snr_sweep.py └── analyze_constellation.py

---

## How to Run

```bash
pip install -r requirements.txt
python examples/run_v1_demo.py
python examples/run_v2_demo.py


---

Citation

Riley, J.H. (2025). ELMOD-16 v1.0: Evolutionary Light Modulation Discovery.
Independent Researcher (UK).


---

License

Code: MIT License (© 2025 Jake Harry Riley)

Document: CC-BY 4.0 (Attribution Required)
