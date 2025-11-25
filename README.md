# ELMOD-16 v1.0  
## Evolutionary Light Modulation Discovery Framework (16-Point)

**Author:** Jake Harry Riley  
**Independent Researcher (UK)**  
**Created:** 25 November 2025  

---

## Overview

ELMOD-16 is an evolutionary research framework for discovering novel digital modulation formats using AI-assisted optimisation.

It includes:

- **V1:** AWGN evolutionary engine  
- **V2:** FTN/ISI + MLSE-lite evolutionary engine  
- **Minimal example scripts for reproducible results**  
- **SNR sweep and constellation analysis tools**  
- **A full v1.0 research specification document (PDF)**  

The framework explores orthogonal and non-orthogonal signalling regimes and is designed for scientific reproducibility and extension.

---

## Key Features

### **V1 – AWGN Engine**
- Evolutionary search over amplitude, phase, roll-off, packing  
- RRC pulse shaping  
- Matched filter + symbol-by-symbol detection  
- Metrics: spectral efficiency, BER, PAPR  
- Evolutionary loop with elitism + mutation  

### **V2 – FTN/ISI + MLSE-Lite Engine**
- Sub-Nyquist packing (ρ < 1.0)  
- Controlled ISI via reduced symbol spacing  
- Approximate channel taps from pulses  
- Light Viterbi detector for ISI regime  
- Complexity-aware fitness penalties  

### **Analysis Tools**
- BER vs SNR curves (via example scripts)  
- Constellation clustering / classifier  
- PAPR and spectral efficiency tools  

---

## Repository Structure

ELMOD-16/ │ ├── README.md ├── LICENSE ├── requirements.txt │ ├── docs/ │   └── ELMOD-16_v1.0_Spec.pdf │ ├── elmod16/ │   ├── init.py │   ├── config.py │   ├── utils.py │   ├── v1_engine.py │   ├── v2_engine.py │   └── analysis.py │ └── examples/ ├── init.py ├── run_v1_demo.py ├── run_v2_demo.py └── snr_sweep.py

---

## How to Run

```bash
pip install -r requirements.txt
python examples/run_v1_demo.py
python examples/run_v2_demo.py


---

Citation

Riley, J.H. (2025). ELMOD-16 v1.0: Evolutionary Light  
Modulation Discovery Framework. Independent Researcher (UK).


---

License

Code: MIT License © 2025 Jake Harry Riley
Documentation: CC-BY 4.0 (Attribution Required)
