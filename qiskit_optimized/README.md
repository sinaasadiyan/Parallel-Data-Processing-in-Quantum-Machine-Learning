# Optimized Qiskit implementation

A drop-in-equivalent, dramatically faster reimplementation of the Method 1 /
Method 2 circuits and cost functions from [`../Parallel_QML.ipynb`](../Parallel_QML.ipynb).
Same model, same math, same results -- just fast. Nothing in the original
notebook was changed; this is a separate, independent implementation.

## Why the original is slow

Three compounding issues in `qc1()`/`qc2()`/`make_backend()`:

1. **Everything is rebuilt from scratch on every single cost evaluation.**
   `classification_probability_1` constructs a brand-new `QuantumCircuit`
   (`ZZFeatureMap`, `TwoLocal`, `.assign_parameters()`, `.to_gate()`) for
   every one of the 128 training points, then calls `transpile()` on the
   whole batch -- every time, even though only the variational *values*
   change between SPSA iterations, not the circuit's structure. `make_backend()`
   also creates a fresh `AerSimulator()` on every call.
2. **Method 2's "Enc" block is rebuilt every call too**, even though it only
   depends on the (fixed) training data, never on the parameters being
   optimized.
3. **The real killer: `fm_gate.control(num_ctrl_qubits=n1)` has no ancilla
   qubits to work with.** `ZZFeatureMap` isn't a primitive gate with a known
   efficient controlled-version, so Qiskit falls back to generic
   multi-controlled-unitary synthesis, whose gate count blows up steeply with
   `n1`. Measured in [`benchmark.py`](benchmark.py):

   | controls (`n1`) | training points | transpiled gate count |
   |---|---|---|
   | 2 | 4   | 588 |
   | 3 | 8   | 5,004 |
   | 4 | 16  | 14,348 |
   | 7 | 128 (the real case) | *(infeasible to even finish transpiling)* |

   This happens per data point, per re-upload layer, per cost evaluation,
   with no caching.

## The fix

[`qc_core.py`](qc_core.py) implements `Method1Classifier`/`Method2Classifier`
with the same constructor/cost/`performance_evaluation` shape as the
PennyLane port in [`../pennylane_implementation`](../pennylane_implementation),
but built around three changes:

1. Use Qiskit's own `ZZFeatureMap`/`TwoLocal` + `quantum_info.Operator` to
   derive the small (4x4) unitary matrices once -- this *is* what those
   circuits compute, just extracted as a matrix instead of re-executed as a
   circuit every time.
2. **Cache the per-data-point encoder matrices.** They never change across
   SPSA iterations (only the data does, and the data is fixed for a training
   run), so they're computed once and reused. Only the small variational
   `TwoLocal` matrix is recomputed each iteration (a handful of 4x4
   `Operator` calls instead of ~128-512 full circuit constructions +
   transpilations).
3. **Apply those matrices via batched NumPy (`einsum`), skipping `transpile()`
   and `AerSimulator` entirely** for the repeated/hot-loop part. Method 2's
   "controlled application to branch `i` only" is implemented as ordinary
   batched matrix-vector multiplication over a `(num_branches, ansatz_dim)`
   array -- mathematically identical to what the multi-controlled-unitary
   gate does, with none of the transpiler's combinatorial synthesis cost.

## Measured results

From [`benchmark.py`](benchmark.py) (`checkerboard`/`dataset1`, same data and
parameters fed to both implementations):

```
=== Method 1: original vs optimized, full 128-point training set ===
  original   : 10.692s/evaluation   cost=1.004129
  optimized  : 0.039s/evaluation (cache warm)   cost=1.004129
  cost match : diff=4.88e-15
  speedup    : 274x

=== Method 2: optimized, full 128-point training set ===
  optimized  : 0.017s/evaluation (cache warm)   cost=0.837226
  -> a full 200-SPSA-iteration training run (3 cost evals/iteration) takes ~10s
```

A full 200-iteration SPSA training run (using the *same*
`qiskit_machine_learning.optimizers.SPSA` as the original notebook, just with
this faster cost function) completes in **~12 seconds** for either method, vs.
an estimated well over an hour for Method 1 and an infeasible amount of time
for Method 2 with the original implementation.

[`verify_against_original.py`](verify_against_original.py) additionally
confirms 100% prediction agreement and an exact cost match against the
parameters the original notebook actually trained, across 4 dataset families.

## Multiclass counterpart

The same three fixes are also applied to the multiclass notebook in
[`../multiclass/qiskit_optimized/`](../multiclass/qiskit_optimized) (4-qubit
ansatz, 4-way classification) -- see that folder's README for its own
measured numbers.

## Files

- [`Parallel_QML_Optimized.ipynb`](Parallel_QML_Optimized.ipynb) — a notebook
  mirroring the structure of [`../Parallel_QML.ipynb`](../Parallel_QML.ipynb)
  (same configuration cell, same `MODE = "load"/"train"` toggle), built on top
  of `qc_core.py`. `MODE = "train"` now actually finishes in ~12 seconds
  instead of being impractical to run from a notebook.
- [`qc_core.py`](qc_core.py) — the optimized classifiers.
- [`benchmark.py`](benchmark.py) — reproduces the timing numbers above
  (also re-run live as the notebook's last cell).
- [`verify_against_original.py`](verify_against_original.py) — correctness
  check against the original notebook's saved, trained parameters.

## Environment

Run with the project's `qiskit-1-4-2` conda environment (same one used for
the PennyLane work) -- it already has Qiskit 1.4.5 and
`qiskit_machine_learning` installed.
