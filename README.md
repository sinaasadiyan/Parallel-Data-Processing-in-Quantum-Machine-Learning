# Parallel Data Processing in Quantum Machine Learning"

A reproducible quantum-machine-learning experiment that classifies four synthetic
2-D datasets with a variational quantum circuit (a **ZZFeatureMap** encoder + a
**TwoLocal** ansatz, re-uploaded 4×, trained with **SPSA**).

📄 Paper: [Parallel Data Processing in Quantum Machine Learning](https://arxiv.org/abs/2508.12006)
(arXiv:2508.12006) — M. Ramezani, S. Asadiyan Zargar, A. Bahrampour, S. Bagheri Shouraki, A. Bahrampour.

Everything lives in a **single notebook** — [`Parallel_QML.ipynb`](QML_TwoLocal_Classifier.ipynb).
Pick a dataset, run all cells, and the results are displayed inline.

## Datasets

Four dataset families, each with **10 independent instances** (`dataset1` … `dataset10`):

| Family         | Description                                  |
| -------------- | -------------------------------------------- |
| `checkerboard` | Alternating checkerboard pattern             |
| `circles`      | Concentric circular regions                  |
| `corners`      | Class regions in opposite corners            |
| `semicircles`  | Two interleaving half-moons                  |

Each instance has 128 training and 64 test points (`feature 1`, `feature 2`) with
binary labels, under [`datasets/<family>/dataset<N>/`](datasets).

## How to use

1. Open `Parallel_QML.ipynb`.
2. In the **Configuration** cell (the only one you normally edit) choose:
   - `DATASET_TYPE` — `"checkerboard"`, `"circles"`, `"corners"`, or `"semicircles"`
   - `DATASET_NUM`  — an instance from `1` to `10`
   - `MODE` — `"load"` to display the pre-computed results (fast, no training),
     or `"train"` to reproduce a run from scratch with SPSA
   - `METHOD` — `1` (one circuit per sample) or `2` (all samples in superposition)
3. **Run All.** The notebook then shows, inline:
   - the raw train/test data,
   - the encoder / variational / full circuit diagrams,
   - training curves (loss, train/test accuracy, SPSA step size),
   - the learned decision boundary and per-point test predictions,
   - a final performance summary,
   - an optional comparison across all 10 instances of the selected family.

## The two methods

- **Method 1** builds a separate circuit for each data point — the standard
  per-sample variational classifier.
- **Method 2** encodes the whole training set in superposition through an index
  register (a QRAM-style batch encoding) and evaluates the cost in a single
  circuit. (For Method 2 a GPU Aer backend can be enabled via `USE_GPU = True`;
  it automatically falls back to CPU if no GPU is available.)

In both cases, train/test **accuracy is measured with the Method-1 read-out**, so
parameters trained by either method are evaluated consistently.

## Pre-computed results

`results/<family>/dataset<N>/method{1,2}.json` contain the full training history
(loss, accuracies, step sizes, and the final optimized parameters) for every
dataset and both methods. These power `MODE = "load"` and the cross-instance
comparison, so reviewers can inspect the outcomes without re-running anything.

## Multiclass extension

The [`multiclass/`](multiclass) folder contains a second, independent experiment:
a **4-class** checkerboard dataset (`checkerboard-4L-4Q`), again in **10
instances**, classified with a variant of the same circuit — a **4-qubit**
TwoLocal ansatz (2 encoding qubits + 2 extra readout qubits, giving a 2-bit
label readout for the 4 classes), re-uploaded **2×**, also trained with SPSA
and following the same Method 1 / Method 2 split.

It follows the same "single notebook, `load`/`train` toggle" pattern as the
root notebook:

- [`multiclass/Parallel_QML_Checkerboard4L.ipynb`](multiclass/Parallel_QML_Checkerboard4L.ipynb)
  — pick a dataset instance, `MODE = "load"` or `"train"`, *Run All*. Must be
  run with its working directory set to `multiclass/`.
- [`multiclass/datasets/dataset<N>/`](multiclass/datasets) — train/test data
  and 4-class labels.
- [`multiclass/results/checkerboard-4L-4Q-dataset<N>/`](multiclass/results)
  — saved SPSA training history per method (powers `MODE = "load"`).
- [`multiclass/checkerboard-4L-4Q-dataset1.ipynb`](multiclass/checkerboard-4L-4Q-dataset1.ipynb)
  — the original training notebook used to produce the saved results.
- `multiclass/auto_results_analyzer_*.py` — standalone scripts that aggregate
  accuracy/cost across all 10 instances into summary plots and reports.

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     |     Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook Parallel_QML.ipynb
```

> **Qiskit version note:** the training code uses `qiskit-machine-learning`'s SPSA
> optimizer, which requires **Qiskit < 2.0** (pinned in `requirements.txt`).
> `MODE = "load"` works on newer Qiskit as well.

## Repository layout

```
QML_TwoLocal_Classifier.ipynb   # the single notebook (the only file users run)
datasets/<family>/dataset<N>/   # train/test data and labels
results/<family>/dataset<N>/    # method1.json, method2.json (saved runs)
requirements.txt

multiclass/                                       # 4-class checkerboard extension
multiclass/Parallel_QML_Checkerboard4L.ipynb       # load/train notebook for this variant
multiclass/datasets/dataset<N>/                    # train/test data and 4-class labels
multiclass/results/checkerboard-4L-4Q-dataset<N>/  # saved SPSA training history per method
multiclass/checkerboard-4L-4Q-dataset1.ipynb        # original training notebook
multiclass/auto_results_analyzer_*.py               # cross-instance summary scripts
```
