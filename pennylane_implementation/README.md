# PennyLane re-implementation

A PennyLane port of the two circuit constructions from
[`../Parallel_QML.ipynb`](../Parallel_QML.ipynb) (originally written with Qiskit).
The `multiclass/` experiment is out of scope here.

## Files

- [`Parallel_QML_PennyLane.ipynb`](Parallel_QML_PennyLane.ipynb) — the notebook,
  mirroring the structure of the original (configuration → load data → circuit
  diagrams → train/load → curves → predictions → summary).
- [`qml_core.py`](qml_core.py) — the PennyLane circuits:
  - `Method1Classifier` — one circuit per data point.
  - `Method2Classifier` — the whole training set encoded in superposition via
    an index register (QRAM-style batch encoding).
- [`train.py`](train.py) — a small, framework-agnostic SPSA optimizer (PennyLane's
  built-in `SPSAOptimizer` expects autograd-tracked arrays and array-valued
  costs; gradient-free SPSA is simple enough to implement directly).
- [`verify_against_qiskit.py`](verify_against_qiskit.py) — loads the
  Qiskit-trained parameters shipped in [`../results`](../results) and confirms
  the PennyLane circuits reproduce **bit-identical predictions**.

## Gate-level equivalence

Both circuits were reverse-engineered from Qiskit's `ZZFeatureMap` (`reps=2`,
full entanglement) and `TwoLocal(['ry','rz'], 'cz', 'full', reps=1)`
decompositions, gate-by-gate. Run the verification script to confirm:

```bash
python verify_against_qiskit.py
```

It reports 100% prediction agreement and an exact final-cost match against the
saved Qiskit results for both methods.

## Method 2's "hidden" labels

Method 2 never receives `train_labels` directly. It works because the training
data is pre-sorted (all label-0 points first, all label-1 points last) and the
dataset size is a power of two (128 = 2^7). The top bit of the index register
then equals the true label, so a circuit that flips a label qubit whenever
(top index bit, predicted qubit) agree is equivalent to measuring the
probability of a correct batch prediction. This exactly mirrors `qc2()` in the
original notebook and is why Method 2 requires `len(train_data)` to be a power
of two with this specific ordering.

## Environment

Developed and executed against the project's `qiskit-1-4-2` conda environment
(already has PennyLane 0.40 + PennyLane-Lightning installed alongside Qiskit
1.4.5, matplotlib, pandas, jupyter). `Method2Classifier` automatically prefers
the faster `lightning.qubit` device and falls back to `default.qubit` if
unavailable.

## Performance notes

- Method 1 batches the whole dataset into a single circuit execution via
  PennyLane parameter broadcasting (~100x faster than one call per sample).
- Method 2's batch circuit precomputes each data point's 2-qubit `ZZFeatureMap`
  unitary as a dense matrix and applies it as a single `ControlledQubitUnitary`
  (7 controls), instead of decomposing every encoder gate individually —
  several orders of magnitude faster than the naive approach.
- Live SPSA training (`MODE = "train"`) is fast for Method 1 (seconds) but
  slow for Method 2 (~4s/circuit evaluation, 3 evaluations/iteration). Reduce
  `MAXITER` for a quicker run.
