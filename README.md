# Perturbation Bounds for Bracket-Closed Subspaces

**Stability and Identifiability of Matrix Lie Algebra Recovery from Noisy Data**

Sooraj K.C and Vivek Mishra  
Department of Pure and Applied Mathematics, Alliance University, Bangalore, India

Submitted to *Linear Algebra and its Applications* (LAA).

---

## Overview

This repository contains the code to reproduce all experiments in the paper. We study the perturbation-theoretic properties of recovering a matrix Lie subalgebra from noisy transformation data — a structured subspace estimation problem where the subspace must be closed under the matrix commutator bracket.

### Main Results

| Result | Theorem | Description |
|--------|---------|-------------|
| Structured subspace perturbation | Thm 4.15 | Lipschitz stability of structure constants via Davis–Kahan sin Θ bounds |
| Local diagnostic validity | Thm 4.1 | Small closure defect certifies proximity to exact Lie algebra |
| Non-identifiability | Thm 3.2 | Recovery only up to conjugacy class |
| Distinguishability lower bound | Thm 4.18 | Information-theoretic limit via Le Cam's method |

## Experiments

All five experiments are implemented in a single file `PerturbationBounds_experiments.py`:

| Experiment | Paper Reference | What it validates |
|------------|----------------|-------------------|
| **Exp 1**: Stability analysis | Table 1, Figure 1 | d_Grass = O(δ) with slope 1.0 for so(16) |
| **Exp 2**: Spectral gap dependence | Table 2, Figure 2 | d_SC = O(δ/γ) for so(8) |
| **Exp 3a**: Binary classification | Table 3, Figure 3 | so(3) vs sl(2,ℝ) via Killing form |
| **Exp 3b**: Multi-class selection | Tables 4–5 | Identify so(d) among {so, sl, sp} |
| **Exp 4**: Falsifiability diagnostic | Table 6, Figure 4 | Closure defect + ROC (AUC = 1.000) |
| **Exp 5**: Sensitivity analysis | Table 7, Figure 5 | Semisimple rigidity vs solvable fragility |

## Quick Start

```bash
# Clone
git clone https://github.com/soorajkcphd/PerturbationBoundsforBracket-ClosedSubspaces.git
cd PerturbationBoundsforBracket-ClosedSubspaces

# Install dependencies
pip install -r requirements.txt

# Run all experiments (~13 min on CPU, ~8 min with GPU)
python PerturbationBounds_experiments.py

# Run specific experiments
python PerturbationBounds_experiments.py 1 3      # only Exp 1 and Exp 3
python PerturbationBounds_experiments.py --cpu    # force CPU mode
```

## Output

The script produces:
- **Tables 1–7** printed to stdout (matching the paper exactly)
- **Figures 1–6** saved as PNG files:
  - `exp1_stability.png` — Recovery error vs noise level
  - `exp2_spectral_gap.png` — Structure constant error vs spectral gap
  - `exp3_phase_transition.png` — Classification phase transition
  - `exp4_model_selection.png` — Multi-class model selection
  - `exp5_sensitivity.png` — Rigidity vs fragility
  - `exp6_falsifiability.png` — ROC curve for falsifiability diagnostic

## Requirements

- Python ≥ 3.10
- NumPy ≥ 1.24
- SciPy ≥ 1.11
- Matplotlib ≥ 3.7

**Optional** (for GPU acceleration and ROC curves):
- PyTorch ≥ 2.0 (with CUDA for GPU batch matrix exponentials)
- scikit-learn ≥ 1.3 (for ROC/AUC computation in Exp 4)

See `requirements.txt` for exact versions.

## Project Structure

```
.
├── PerturbationBounds_experiments.py
├── requirements.txt      # Python dependencies
├── README.md

```

## Citation

If you use this code, please cite:

```bibtex
@article{kc2025perturbation,
  title={Perturbation Bounds for Bracket-Closed Subspaces: Stability and 
         Identifiability of Matrix Lie Algebra Recovery from Noisy Data},
  author={K.C, Sooraj and Mishra, Vivek},
  journal={Linear Algebra and its Applications},
  year={2026},
  note={Submitted}
}
```

## License

MIT License. See `LICENSE` for details.

