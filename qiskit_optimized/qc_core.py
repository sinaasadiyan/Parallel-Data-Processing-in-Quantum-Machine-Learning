"""
A drop-in-equivalent, dramatically faster reimplementation of the Method 1 /
Method 2 circuits from ``../Parallel_QML.ipynb``.

The original notebook is slow to train for three compounding reasons (see
``README.md`` for the measured numbers):

  (a) `qc1()`/`qc2()` rebuild a brand-new Qiskit circuit -- including
      `ZZFeatureMap(...)`, `TwoLocal(...)`, `.assign_parameters(...)`,
      `.to_gate()` -- from scratch on *every single cost evaluation*, even
      though the data-encoding part never changes between SPSA iterations
      (only the variational parameters do).
  (b) `make_backend()` creates a fresh `AerSimulator()` every call, and
      `classification_probability_{1,2}` calls `transpile()` on the whole
      circuit batch every time, even though only the bound parameter VALUES
      change between iterations, not the circuit STRUCTURE.
  (c) Method 2's controlled encoder (`fm_gate.control(num_ctrl_qubits=n1)`)
      asks Qiskit to synthesize a generic n1-controlled arbitrary unitary
      with no ancilla qubits available. Without a known efficient
      controlled-version of `ZZFeatureMap`, this falls back to a generic
      synthesis whose gate count blows up steeply with `n1` (confirmed
      empirically in `benchmark.py`: transpiled gate count goes from ~600 at
      n1=2 to ~14,000 at n1=4 -- at the real n1=7 this is many orders of
      magnitude worse), repeated for every one of the ~128 training points,
      at every re-upload layer, on every single cost evaluation.

The fix applied here, for both methods:
  1. Use Qiskit's own `ZZFeatureMap`/`TwoLocal` + `quantum_info.Operator` to
     derive the small (4x4) unitary matrices -- this guarantees the matrices
     are *exactly* what the original notebook's circuits implement (verified
     against the saved Qiskit-trained parameters in `benchmark.py`).
  2. Cache the per-data-point ZZFeatureMap matrices ONCE (they never change
     across SPSA iterations -- only the data does, and the data is fixed for
     a training run). Only the small variational TwoLocal matrix is rebuilt
     each iteration (a handful of 4x4 Operator calls instead of ~128-512
     full circuit constructions + transpilations).
  3. Apply those cached matrices directly via batched NumPy
     (`numpy.einsum`), bypassing Qiskit's circuit object model AND its
     `transpile()` step entirely for the repeated/hot-loop part. Method 2's
     "controlled application to branch i only" is implemented as ordinary
     batched matrix-vector multiplication over a
     (num_branches, ansatz_dim) array -- mathematically identical to what
     the multi-controlled-unitary gate would do, with none of the
     transpiler's combinatorial gate-synthesis cost.
"""
import itertools

import numpy as np
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit.quantum_info import Operator


# --------------------------------------------------------------------------
# Qiskit-derived (and Qiskit-validated) building-block matrices
# --------------------------------------------------------------------------

def zz_feature_map_matrix(data, n_features=2, reps=2):
    """The exact unitary Qiskit's ZZFeatureMap(data) implements, as a dense matrix."""
    enc = ZZFeatureMap(feature_dimension=n_features, reps=reps)
    enc.assign_parameters(dict(zip(enc.parameters, data)), inplace=True)
    return Operator(enc).data


def two_local_matrix(values, n_qubits, reps=1):
    """The exact unitary Qiskit's TwoLocal(['ry','rz'], 'cz', 'full', reps) implements."""
    var = TwoLocal(n_qubits, ["ry", "rz"], "cz", "full", reps=reps)
    var.assign_parameters(dict(zip(var.parameters, values)), inplace=True)
    return Operator(var).data


def two_local_num_params(n_qubits, reps=1):
    return 2 * n_qubits * (reps + 1)


# --------------------------------------------------------------------------
# Method 1: one circuit per data point -- but evaluated as batched NumPy
# --------------------------------------------------------------------------

class Method1Classifier:
    """Per-sample variational classifier, numerically identical to `qc1()` /
    `classification_probability_1()` in the original notebook, but with the
    per-data-point ZZFeatureMap matrix cached once and the whole batch
    evaluated via NumPy instead of rebuilding+transpiling 128 Qiskit circuits
    on every cost evaluation.
    """

    def __init__(self, n_features, num_reupload):
        self.n_features = n_features
        self.num_reupload = num_reupload
        self.dim = 2 ** n_features
        self.block_params = two_local_num_params(n_features, reps=1)
        self.num_parameters = num_reupload * self.block_params
        self._cache = {}  # id(data) -> cached (N, dim, dim) encoder matrix stack

    def _encoder_stack(self, all_data):
        key = id(all_data)
        cached = self._cache.get(key)
        if cached is not None and cached[0] is all_data:
            return cached[1]
        E = np.stack([zz_feature_map_matrix(d, self.n_features) for d in all_data])
        self._cache[key] = (all_data, E)
        return E

    def classification_probability(self, all_data, variational):
        """Returns (N, dim) probabilities over the readout qubit's basis states."""
        all_data = np.asarray(all_data, dtype=float)
        E = self._encoder_stack(all_data)  # (N, dim, dim), cached across calls
        N = len(all_data)
        psi = np.zeros((N, self.dim), dtype=complex)
        psi[:, 0] = 1.0
        for i in range(self.num_reupload):
            V = two_local_matrix(
                variational[i * self.block_params:(i + 1) * self.block_params],
                self.n_features,
            )
            psi = np.einsum("nij,nj->ni", E, psi)
            psi = np.einsum("ij,nj->ni", V, psi)
        probs = np.abs(psi) ** 2  # (N, dim)

        # Qiskit's label_probability_1() keeps qubit 0 only (traces out the
        # rest): qubit 0 is the LSB of the basis-state index.
        p0 = probs[:, 0::2].sum(axis=1)
        p1 = probs[:, 1::2].sum(axis=1)
        return np.stack([p0, p1], axis=1)

    def predict(self, all_data, variational):
        probs = self.classification_probability(all_data, variational)
        return list(np.argmax(probs, axis=1))

    def performance_evaluation(self, data, labels, variational):
        probs = self.classification_probability(data, variational)
        predictions = list(np.argmax(probs, axis=1))
        accuracy = np.mean([int(pr == lb) for pr, lb in zip(predictions, labels)])
        return accuracy, predictions

    def cost(self, all_data, labels, variational):
        probs = self.classification_probability(all_data, variational)
        losses = [-np.log(max(probs[i, int(lb)], 1e-12)) for i, lb in enumerate(labels)]
        return float(np.mean(losses))


# --------------------------------------------------------------------------
# Method 2: whole dataset in superposition -- batched NumPy, no transpile
# --------------------------------------------------------------------------

class Method2Classifier:
    """Batch classifier, numerically identical to `qc2()` /
    `classification_probability_2()` in the original notebook.

    Instead of synthesizing a 7-controlled `ZZFeatureMap` gate (the main cost
    driver -- see the module docstring), each training point's cached 4x4
    encoder matrix is applied directly to its own "branch" of a
    (num_data, ansatz_dim) array -- exactly what the controlled-unitary gate
    does mathematically, with none of the generic-synthesis overhead.
    Requires the same things `qc2()` does: `len(all_data)` a power of two,
    sorted with all label-0 points first and all label-1 points last.
    """

    def __init__(self, n_features, num_reupload, num_data):
        self.n_features = n_features
        self.num_reupload = num_reupload
        self.num_data = num_data
        self.dim = 2 ** n_features
        self.n_index = int(round(np.log2(num_data)))
        if 2 ** self.n_index != num_data:
            raise ValueError("Method 2 requires the dataset size to be a power of two.")

        self.block_params = two_local_num_params(n_features, reps=1)
        self.num_parameters = num_reupload * self.block_params
        self._cache = {}

    def _encoder_stack(self, all_data):
        key = id(all_data)
        cached = self._cache.get(key)
        if cached is not None and cached[0] is all_data:
            return cached[1]
        E = np.stack([zz_feature_map_matrix(d, self.n_features) for d in all_data])
        self._cache[key] = (all_data, E)
        return E

    def label_probability(self, all_data, variational):
        all_data = np.asarray(all_data, dtype=float)
        N = len(all_data)
        E = self._encoder_stack(all_data)  # (N, dim, dim), cached across calls

        psi = np.zeros((N, self.dim), dtype=complex)
        psi[:, 0] = 1.0 / np.sqrt(N)  # uniform index-register superposition folded in
        for i in range(self.num_reupload):
            V = two_local_matrix(
                variational[i * self.block_params:(i + 1) * self.block_params],
                self.n_features,
            )
            psi = np.einsum("nij,nj->ni", E, psi)
            psi = np.einsum("ij,nj->ni", V, psi)
        probs = np.abs(psi) ** 2  # (N, dim)

        # The data is sorted (all label-0 first, all label-1 last), so the top
        # index bit equals the true label; the predicted label is qubit 0 of
        # the ansatz state (its LSB). p1 = P(top index bit == predicted qubit).
        n_idx = np.arange(N)
        true_bit = (n_idx >= N // 2).astype(int)
        a_idx = np.arange(self.dim)
        pred_bit = a_idx & 1
        match = true_bit[:, None] == pred_bit[None, :]
        p1 = float(probs[match].sum())
        return {"0": 1.0 - p1, "1": p1}

    def cost(self, all_data, variational):
        p = self.label_probability(all_data, variational)
        return float(-np.log(max(p.get("1", 0), 1e-12)))
