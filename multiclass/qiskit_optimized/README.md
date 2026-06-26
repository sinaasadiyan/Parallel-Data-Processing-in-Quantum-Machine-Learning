# Optimized Qiskit implementation (multiclass)

A drop-in-equivalent, dramatically faster reimplementation of the Method 1 /
Method 2 circuits from
[`../Parallel_QML_Checkerboard4L.ipynb`](../Parallel_QML_Checkerboard4L.ipynb).
This is the multiclass counterpart of
[`../../qiskit_optimized/`](../../qiskit_optimized) -- same fixes, generalized
to the 4-qubit ansatz (2 encoding qubits + 2 extra "workspace" qubits) and
4-way classification.

## The fix (same three issues, see [`../../qiskit_optimized/README.md`](../../qiskit_optimized/README.md) for the full writeup)

1. **Cache each data point's ZZFeatureMap matrix once** (it never changes
   across SPSA iterations), instead of rebuilding the encoder circuit from
   scratch on every cost evaluation.
2. **Derive matrices via Qiskit's own `ZZFeatureMap`/`TwoLocal` +
   `quantum_info.Operator`** (guaranteeing exact correctness), then apply
   them with batched NumPy instead of `transpile()` + `AerSimulator`.
3. **For Method 2, apply each data point's encoder matrix directly** to its
   own "branch" of the batch array, instead of letting Qiskit synthesize a
   7-controlled arbitrary unitary with no ancilla qubits.

One addition specific to the multiclass case: the encoder only acts on the
first `n_features` ("encoding") qubits of the `2 * n_features`-qubit ansatz
register; the extra "workspace" qubits are left untouched by the encoder.
Embedding the small encoder matrix into the full ansatz space is a Kronecker
product `np.kron(I_workspace, E_small)` -- verified directly against Qiskit's
own `Operator` of the equivalent circuit (workspace qubits have higher qubit
index, hence are the more-significant factor in Qiskit's little-endian
convention; see the comment at the top of `qc_core.py`).

## Measured results

From [`benchmark.py`](benchmark.py) (`dataset1`, same data and parameters fed
to both implementations):

```
=== Method 1: original vs optimized, full 128-point training set ===
  original   : 6.56s/evaluation   cost=1.563736
  optimized  : 0.014s/evaluation (cache warm)   cost=1.563736
  cost match : diff=2.00e-15
  speedup    : 458x

=== Method 2: optimized, full 128-point training set ===
  optimized  : 0.012s/evaluation (cache warm)   cost=1.344014
  -> a full 400-SPSA-iteration training run (3 cost evals/iteration) takes ~15s
```

A full 400-iteration SPSA training run (using the *same*
`qiskit_machine_learning.optimizers.SPSA` as the original notebook) completes
in **~15-20 seconds** for either method.

[`verify_against_original.py`](verify_against_original.py) confirms 100%
prediction agreement and an exact cost match against the parameters the
original notebook actually trained, across 4 dataset instances.

## Files

- [`Parallel_QML_Checkerboard4L_Optimized.ipynb`](Parallel_QML_Checkerboard4L_Optimized.ipynb) —
  a notebook mirroring the structure of
  [`../Parallel_QML_Checkerboard4L.ipynb`](../Parallel_QML_Checkerboard4L.ipynb),
  built on top of `qc_core.py`. `MODE = "train"` now actually finishes in
  ~15-20 seconds instead of being impractical to run from a notebook.
- [`qc_core.py`](qc_core.py) — the optimized classifiers.
- [`benchmark.py`](benchmark.py) — reproduces the timing numbers above
  (also re-run live as the notebook's last cell).
- [`verify_against_original.py`](verify_against_original.py) — correctness
  check against the original notebook's saved, trained parameters.

## A Windows path-length gotcha

Same issue as [`../pennylane_implementation`](../pennylane_implementation)
hit: this folder is nested two levels deep
(`multiclass/qiskit_optimized/`), so un-resolved relative paths like
`os.path.join("..", "results", ...)` can exceed Windows' 260-char `MAX_PATH`
limit for some file APIs. The notebook uses `os.path.abspath(...)` (not just
`os.path.normpath`) when building paths that cross into `../results` or
`../datasets`.

## Environment

Run with the project's `qiskit-1-4-2` conda environment (same one used
everywhere else in this repo) -- it already has Qiskit 1.4.5 and
`qiskit_machine_learning` installed.
