# ELMOD-16 v1.0  
## Evolutionary Light Modulation Discovery Framework (16-Point)

**Author:** Jake Harry Riley  
**Independent Researcher (UK)**  
**Created:** 25 November 2025  

---

## Overview
**Licensing:** ELMOD-16 is dual-licensed: free for research and academic use under CC BY-NC 4.0, and available for commercial licensing upon request.

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

Riley, J.H. (2025). ELMOD-16 v1.0: Evolutionary Light Modulation Discovery Framework (16-Point).
Independent Researcher (UK).
GitHub: https://github.com/JRiley90/ELMOD-16


---

License

Code License (MIT):
The source code of ELMOD-16 v1.0 is released under the MIT License, permitting reuse, modification, and distribution with attribution.

Documentation & Research Specification (CC BY-NC 4.0):
The documentation, research specification PDF, and descriptive materials are available for non-commercial use with attribution.

Commercial Licensing:
Commercial use of any part of ELMOD-16 (code, documentation, research outputs, or derivative works) requires a commercial license agreement.
Contact: jakeriley407@gmail.com
