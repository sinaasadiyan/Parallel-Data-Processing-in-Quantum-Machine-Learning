# PennyLane re-implementation (multiclass)

A PennyLane port of the two circuit constructions from
[`../Parallel_QML_Checkerboard4L.ipynb`](../Parallel_QML_Checkerboard4L.ipynb)
(originally written with Qiskit). This is the multiclass counterpart of
[`../../pennylane_implementation/`](../../pennylane_implementation) -- same
recipe, generalized to a 4-qubit ansatz and 4-way classification.

## Files

- [`Parallel_QML_Checkerboard4L_PennyLane.ipynb`](Parallel_QML_Checkerboard4L_PennyLane.ipynb) —
  the notebook, mirroring the structure of the original.
- [`qml_core.py`](qml_core.py) — the PennyLane circuits:
  - `Method1Classifier` — one circuit per data point. The ansatz acts on
    `2 * n_features = 4` qubits (2 "encoding" qubits, re-fed by the
    ZZFeatureMap at every re-upload, + 2 extra "workspace" qubits that only
    see the TwoLocal ansatz). Classification reads out the joint 2-bit state
    of the *encoding* qubits as one of 4 classes; the workspace qubits are
    discarded.
  - `Method2Classifier` — the whole training set encoded in superposition via
    an index register, generalized to 4 classes: the training data must be
    pre-sorted into 4 contiguous, equal-size blocks (class 0, 1, 2, 3 in
    order), so the top 2 bits of the index register equal the true class.
- [`train.py`](train.py) — the same framework-agnostic SPSA optimizer used by
  the binary implementation (copied verbatim; nothing here is binary-specific).
- [`verify_against_qiskit.py`](verify_against_qiskit.py) — loads the
  Qiskit-trained parameters shipped in [`../results`](../results) and confirms
  the PennyLane circuits reproduce **bit-identical predictions**.

## Gate-level equivalence

Both circuits were reverse-engineered from Qiskit's `ZZFeatureMap` (`reps=2`,
2 qubits) and `TwoLocal(['ry','rz'], 'cz', 'full', reps=1)` on **4 qubits**
decompositions. The 4-qubit `TwoLocal` interleaves its CZ entanglers and
rotation gates differently from the 2-qubit case (Qiskit's own construction
algorithm), but since gates on disjoint qubits commute, the simpler
`[rotate][entangle][rotate]` layout used here (same as the 2-qubit case)
produces the exact same unitary -- verified directly against Qiskit's
statevector for random parameters (see the comment in `two_local_block` in
`qml_core.py`).

Run the verification script to confirm end-to-end:

```bash
python verify_against_qiskit.py
```

It reports 100% prediction agreement and an exact final-cost match against the
saved Qiskit results for both methods, across multiple dataset instances.

## Environment

Developed and executed against the project's `qiskit-1-4-2` conda environment
(PennyLane 0.40 + PennyLane-Lightning, alongside Qiskit 1.4.5, matplotlib,
pandas, jupyter).

## A Windows path-length gotcha

This folder is nested two levels deep (`multiclass/pennylane_implementation/`)
under an already-long project path. Un-resolved relative paths like
`os.path.join("..", "results", ...)` can exceed Windows' classic 260-char
`MAX_PATH` limit for some file APIs even when the *normalized* absolute path
is well within it. The notebook and scripts here use `os.path.abspath(...)`
(not just `os.path.normpath`) when building paths that cross into `../results`
or `../datasets`, since only `abspath` actually collapses `..` against `cwd`
into a short, absolute path.

## Performance notes

Same as the binary implementation: Method 1 batches the whole dataset into a
single circuit execution; Method 2 precomputes each point's 2-qubit
`ZZFeatureMap` unitary and applies it as one `ControlledQubitUnitary` (7
controls) instead of decomposing every encoder gate. Live SPSA training
(`MODE = "train"`) is fast for Method 1 but slow for Method 2 (~4s/circuit
evaluation, 3 evaluations/iteration) -- reduce `MAXITER` for a quicker run.
