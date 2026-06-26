# Parallel Data Processing in Quantum Machine Learning

A reproducible quantum-machine-learning experiment that classifies four synthetic
2-D datasets with a variational quantum circuit (a **ZZFeatureMap** encoder + a
**TwoLocal** ansatz, re-uploaded 4×, trained with **SPSA**).

📄 Paper: [Parallel Data Processing in Quantum Machine Learning](https://arxiv.org/abs/2508.12006)
(arXiv:2508.12006) — M. Ramezani, S. Asadiyan Zargar, A. Bahrampour, S. Bagheri Shouraki, A. Bahrampour.

The primary notebooks used to generate the paper's results are
[`Parallel_QML.ipynb`](Parallel_QML.ipynb) (binary) and
[`multiclass/Parallel_QML_Checkerboard4L.ipynb`](multiclass/Parallel_QML_Checkerboard4L.ipynb)
(4-class). Both are implemented with raw Qiskit and require a long execution
time to train from scratch. For faster experimentation, optimized and PennyLane
re-implementations are provided in the subfolders below — same model, same
results, dramatically less waiting.

## Which version should I use?

| Version | Binary notebook | Multiclass notebook | Training time (Method 1) |
|---|---|---|---|
| **Original Qiskit** | `Parallel_QML.ipynb` | `multiclass/Parallel_QML_Checkerboard4L.ipynb` | ≫ 1 hour (estimated) |
| **Optimized Qiskit** ⭐ | `qiskit_optimized/Parallel_QML_Optimized.ipynb` | `multiclass/qiskit_optimized/Parallel_QML_Checkerboard4L_Optimized.ipynb` | ~12 s / ~15 s |
| **PennyLane** | `pennylane_implementation/Parallel_QML_PennyLane.ipynb` | `multiclass/pennylane_implementation/Parallel_QML_Checkerboard4L_PennyLane.ipynb` | seconds (Method 1); slower for Method 2 |

All versions share the same datasets, saved results, and `MODE = "load"/"train"` toggle.

---

## Datasets

Four dataset families, each with **10 independent instances** (`dataset1` … `dataset10`):

| Family | Description |
|---|---|
| `checkerboard` | Alternating checkerboard pattern |
| `circles` | Concentric circular regions |
| `corners` | Class regions in opposite corners |
| `semicircles` | Two interleaving half-moons |

Each instance has 128 training and 64 test points (`feature 1`, `feature 2`) with
binary labels, under [`datasets/<family>/dataset<N>/`](datasets).

---

## The two circuit methods

- **Method 1** builds a separate circuit for each data point — the standard
  per-sample variational classifier.
- **Method 2** encodes the whole training set in superposition through an index
  register (a QRAM-style batch encoding) and evaluates the cost in a single
  circuit. (For Method 2 in the original notebook, a GPU Aer backend can be
  enabled via `USE_GPU = True`; it automatically falls back to CPU if no GPU is
  available.)

In both cases, train/test **accuracy is measured with the Method-1 read-out**, so
parameters trained by either method are evaluated consistently.

---

## How to Run

All notebooks share the same configuration pattern:

1. Open the notebook.
2. Edit the **Configuration** cell:
   - `DATASET_TYPE` — `"checkerboard"`, `"circles"`, `"corners"`, or `"semicircles"`
   - `DATASET_NUM` — an instance from `1` to `10`
   - `MODE` — `"load"` to display pre-computed results (fast, no training), or
     `"train"` to reproduce a run from scratch
   - `METHOD` — `1` or `2`
3. **Run All.** Results are displayed inline (data, circuit diagrams, training
   curves, decision boundary, performance summary, cross-instance comparison).

### Original Qiskit notebooks

```
Parallel_QML.ipynb                                 # binary
multiclass/Parallel_QML_Checkerboard4L.ipynb       # 4-class
```

These are the reference implementations from the paper. `MODE = "load"` is
fast; `MODE = "train"` is slow (the cost function rebuilds and transpiles the
full circuit batch on every SPSA evaluation).

> **Recommended for:** reproducing the exact paper figures, inspecting the
> circuit structure, or running with `MODE = "load"`.

### Optimized Qiskit notebooks ⭐

```
qiskit_optimized/Parallel_QML_Optimized.ipynb
multiclass/qiskit_optimized/Parallel_QML_Checkerboard4L_Optimized.ipynb
```

Drop-in replacements that are mathematically identical to the originals —
same model, same optimizer, same saved results — but train in **seconds**
instead of hours. The speedup comes from three changes (see
[`qiskit_optimized/README.md`](qiskit_optimized/README.md) for the full
explanation):

1. Per-data-point `ZZFeatureMap` matrices are cached once and reused across
   all SPSA iterations.
2. Matrices are derived via `quantum_info.Operator` and applied with batched
   NumPy `einsum`, skipping `transpile()` and `AerSimulator` in the hot loop.
3. Method 2's controlled-encoder is applied by direct branch-wise matrix
   multiplication, avoiding combinatorially expensive multi-controlled-unitary
   synthesis.

**Measured speedup:**

| | Original | Optimized | Speedup |
|---|---|---|---|
| Method 1 per-eval (binary, 128 pts) | 10.69 s | 0.039 s | ~274× |
| Method 1 per-eval (multiclass, 128 pts) | 6.56 s | 0.014 s | ~458× |
| Method 1 full training (binary, 200 iters) | ≥ 71 min (est.) | ~12 s | ≥ ~370× |
| Method 1 full training (multiclass, 400 iters) | ≥ 88 min (est.) | ~15 s | ≥ ~380× |
| Method 2 full training (both) | hours | ~12–19 s | — |

Correctness is verified against the original notebook's trained parameters:
100% prediction agreement, cost match to within `1e-15`.

To run the benchmark or correctness check directly:
```bash
cd qiskit_optimized
python benchmark.py            # timing + scaling numbers
python verify_against_original.py  # 100% agreement check

cd multiclass/qiskit_optimized
python benchmark.py
python verify_against_original.py
```

> **Recommended for:** training from scratch, experimenting with new datasets
> or hyperparameters, and for most everyday use.

### PennyLane notebooks

```
pennylane_implementation/Parallel_QML_PennyLane.ipynb
multiclass/pennylane_implementation/Parallel_QML_Checkerboard4L_PennyLane.ipynb
```

A full re-implementation of both circuit constructions in PennyLane, with the
same notebook structure and `load`/`train` toggle. The circuits were
reverse-engineered gate-by-gate from Qiskit's `ZZFeatureMap` and `TwoLocal`
decompositions and verified to produce bit-identical predictions. Method 1
uses PennyLane's parameter broadcasting for fast batched evaluation; Method 2
applies each data point's pre-computed encoder matrix as a `ControlledQubitUnitary`.

To verify gate-level equivalence against the Qiskit-trained parameters:
```bash
cd pennylane_implementation
python verify_against_qiskit.py   # 100% prediction agreement

cd multiclass/pennylane_implementation
python verify_against_qiskit.py
```

> **Recommended for:** framework comparisons, PennyLane-specific development,
> or if you prefer PennyLane's device/backend ecosystem.

---

## Pre-computed results

`results/<family>/dataset<N>/method{1,2}.json` contain the full training history
(loss, accuracies, step sizes, and the final optimized parameters) for every
dataset and both methods. These power `MODE = "load"` and the cross-instance
comparison, so reviewers can inspect the outcomes without re-running anything.

---

## Multiclass extension

The [`multiclass/`](multiclass) folder contains a second, independent experiment:
a **4-class** checkerboard dataset (`checkerboard-4L-4Q`), again in **10
instances**, classified with a variant of the same circuit — a **4-qubit**
TwoLocal ansatz (2 encoding qubits + 2 extra readout qubits, giving a 2-bit
label readout for the 4 classes), re-uploaded **2×**, also trained with SPSA
and following the same Method 1 / Method 2 split.

- [`multiclass/Parallel_QML_Checkerboard4L.ipynb`](multiclass/Parallel_QML_Checkerboard4L.ipynb)
  — original Qiskit notebook; `load`/`train` toggle, run from `multiclass/`.
- [`multiclass/qiskit_optimized/`](multiclass/qiskit_optimized) — optimized
  Qiskit version (same fixes as above, generalized to 4-qubit ansatz).
- [`multiclass/pennylane_implementation/`](multiclass/pennylane_implementation)
  — PennyLane version.
- [`multiclass/datasets/dataset<N>/`](multiclass/datasets) — train/test data
  and 4-class labels.
- [`multiclass/results/checkerboard-4L-4Q-dataset<N>/`](multiclass/results)
  — saved SPSA training history per method (powers `MODE = "load"`).
- `multiclass/auto_results_analyzer_*.py` — standalone scripts that aggregate
  accuracy/cost across all 10 instances into summary plots and reports.

---

## Installation

```bash
conda activate qiskit-1-4-2   # recommended: the project's own environment
# or
python -m venv .venv
# Windows: .venv\Scripts\activate  |  Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

> **Qiskit version note:** the training code uses `qiskit-machine-learning`'s SPSA
> optimizer, which requires **Qiskit < 2.0** (pinned in `requirements.txt`).
> `MODE = "load"` works on newer Qiskit as well.

---

## Speedup report

A detailed explanation of why the original notebooks are slow, what was changed,
and the measured speedup numbers is available in
[`Qiskit_Training_Speedup_Report.pdf`](Qiskit_Training_Speedup_Report.pdf).

---

## Repository layout

```
Parallel_QML.ipynb                      # original binary notebook (paper reference)
datasets/<family>/dataset<N>/           # train/test data and labels
results/<family>/dataset<N>/            # method1.json, method2.json (saved runs)
requirements.txt
Qiskit_Training_Speedup_Report.pdf      # speedup analysis report

qiskit_optimized/                       # optimized Qiskit binary implementation
  Parallel_QML_Optimized.ipynb          #   load/train notebook (~12 s to train)
  qc_core.py                            #   cached-matrix + NumPy classifiers
  benchmark.py                          #   timing and scaling measurements
  verify_against_original.py            #   100% agreement check vs. original

pennylane_implementation/               # PennyLane binary re-implementation
  Parallel_QML_PennyLane.ipynb          #   load/train notebook
  qml_core.py                           #   PennyLane circuits (Method 1 & 2)
  train.py                              #   framework-agnostic SPSA optimizer
  verify_against_qiskit.py              #   gate-level equivalence check

multiclass/                             # 4-class checkerboard extension
  Parallel_QML_Checkerboard4L.ipynb     #   original Qiskit notebook (paper reference)
  datasets/dataset<N>/                  #   train/test data and 4-class labels
  results/checkerboard-4L-4Q-dataset<N>/#   saved SPSA training history
  auto_results_analyzer_*.py            #   cross-instance summary scripts

  qiskit_optimized/                     #   optimized Qiskit multiclass version
    Parallel_QML_Checkerboard4L_Optimized.ipynb  # (~15 s to train)
    qc_core.py
    benchmark.py
    verify_against_original.py

  pennylane_implementation/             #   PennyLane multiclass re-implementation
    Parallel_QML_Checkerboard4L_PennyLane.ipynb
    qml_core.py
    train.py
    verify_against_qiskit.py
```
