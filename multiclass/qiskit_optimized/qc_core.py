"""
A drop-in-equivalent, dramatically faster reimplementation of the Method 1 /
Method 2 circuits from ``../Parallel_QML_Checkerboard4L.ipynb``.

This is the multiclass (4-class checkerboard) counterpart of
``../../qiskit_optimized/qc_core.py``. Same fixes, same reasoning -- see that
file's docstring and ``../../qiskit_optimized/README.md`` for the full
writeup -- generalized to a 4-qubit ansatz (2 encoding qubits + 2 extra
"workspace" qubits) and 4-way classification:

  1. Cache each data point's ZZFeatureMap matrix once (it never changes
     across SPSA iterations), instead of rebuilding the encoder circuit from
     scratch on every cost evaluation.
  2. Derive matrices via Qiskit's own `ZZFeatureMap`/`TwoLocal` +
     `quantum_info.Operator` (guaranteeing exact correctness), then apply
     them with batched NumPy instead of `transpile()` + `AerSimulator`.
  3. For Method 2, apply each data point's encoder matrix directly to its own
     "branch" of a (num_data, ansatz_dim) array, instead of asking Qiskit to
     synthesize a 7-controlled arbitrary unitary with no ancilla qubits
     (the original's main bottleneck -- see the binary version's benchmark).

One addition specific to the multiclass case: the encoder only acts on the
first `n_features` ("encoding") qubits of the `2 * n_features`-qubit ansatz
register, with the extra "workspace" qubits left untouched. Embedding the
small encoder matrix into the full ansatz space is a Kronecker product
`np.kron(I_workspace, E_small)` -- verified directly against Qiskit's own
`Operator` of the equivalent circuit (workspace qubits have higher qubit
index, hence are the more-significant factor in Qiskit's little-endian
convention).
"""
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
    """Per-sample variational classifier for the 4-class checkerboard task.

    Numerically identical to `qc1()`/`classification_probability_1()` in the
    original notebook: the ansatz acts on `2 * n_features` qubits (the first
    `n_features` "encoding" qubits, re-fed by the ZZFeatureMap at every
    re-upload, + `n_features` extra "workspace" qubits that only ever see the
    TwoLocal ansatz). Classification reads out the *encoding* qubits' joint
    state as an `n_features`-bit class label.
    """

    def __init__(self, n_features, num_reupload):
        self.n_features = n_features
        self.n_ansatz = 2 * n_features
        self.num_reupload = num_reupload
        self.enc_dim = 2 ** n_features
        self.dim = 2 ** self.n_ansatz
        self.n_classes = 2 ** n_features
        self.block_params = two_local_num_params(self.n_ansatz, reps=1)
        self.num_parameters = num_reupload * self.block_params
        self._cache = {}

    def _encoder_stack(self, all_data):
        key = id(all_data)
        cached = self._cache.get(key)
        if cached is not None and cached[0] is all_data:
            return cached[1]
        E_small = [zz_feature_map_matrix(d, self.n_features) for d in all_data]
        I_ws = np.eye(2 ** (self.n_ansatz - self.n_features))
        E_full = np.stack([np.kron(I_ws, e) for e in E_small])  # (N, dim, dim)
        self._cache[key] = (all_data, E_full)
        return E_full

    def classification_probability(self, all_data, variational):
        all_data = np.asarray(all_data, dtype=float)
        E = self._encoder_stack(all_data)  # (N, dim, dim), cached across calls
        N = len(all_data)
        psi = np.zeros((N, self.dim), dtype=complex)
        psi[:, 0] = 1.0
        for i in range(self.num_reupload):
            V = two_local_matrix(
                variational[i * self.block_params:(i + 1) * self.block_params],
                self.n_ansatz,
            )
            psi = np.einsum("nij,nj->ni", E, psi)
            psi = np.einsum("ij,nj->ni", V, psi)
        probs_full = np.abs(psi) ** 2  # (N, dim)

        # Qiskit's label_probability_1() keeps the encoding qubits (the LSBs
        # of the full basis index) and traces out the workspace qubits.
        probs = np.zeros((N, self.enc_dim))
        for c in range(self.enc_dim):
            probs[:, c] = probs_full[:, c::self.enc_dim].sum(axis=1)
        return probs  # shape (N, n_classes)

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

    Requires `len(all_data)` to be a power of two AND the data ordered into
    `n_classes` contiguous, equal-size, increasing blocks (all class-0 points
    first, ..., all class-(n_classes-1) points last) -- the top `n_features`
    bits of the index register then equal the true class.
    """

    def __init__(self, n_features, num_reupload, num_data):
        self.n_features = n_features
        self.n_ansatz = 2 * n_features
        self.num_reupload = num_reupload
        self.num_data = num_data
        self.enc_dim = 2 ** n_features
        self.dim = 2 ** self.n_ansatz
        self.n_classes = 2 ** n_features

        self.n_index = int(round(np.log2(num_data)))
        if 2 ** self.n_index != num_data:
            raise ValueError("Method 2 requires the dataset size to be a power of two.")
        if num_data % self.n_classes != 0:
            raise ValueError("Method 2 requires the dataset size to be divisible by n_classes.")

        self.block_params = two_local_num_params(self.n_ansatz, reps=1)
        self.num_parameters = num_reupload * self.block_params
        self._cache = {}

    def _encoder_stack(self, all_data):
        key = id(all_data)
        cached = self._cache.get(key)
        if cached is not None and cached[0] is all_data:
            return cached[1]
        E_small = [zz_feature_map_matrix(d, self.n_features) for d in all_data]
        I_ws = np.eye(2 ** (self.n_ansatz - self.n_features))
        E_full = np.stack([np.kron(I_ws, e) for e in E_small])  # (N, dim, dim)
        self._cache[key] = (all_data, E_full)
        return E_full

    def label_probability(self, all_data, variational):
        all_data = np.asarray(all_data, dtype=float)
        N = len(all_data)
        E = self._encoder_stack(all_data)  # (N, dim, dim), cached across calls

        psi = np.zeros((N, self.dim), dtype=complex)
        psi[:, 0] = 1.0 / np.sqrt(N)  # uniform index-register superposition folded in
        for i in range(self.num_reupload):
            V = two_local_matrix(
                variational[i * self.block_params:(i + 1) * self.block_params],
                self.n_ansatz,
            )
            psi = np.einsum("nij,nj->ni", E, psi)
            psi = np.einsum("ij,nj->ni", V, psi)
        probs = np.abs(psi) ** 2  # (N, dim)

        # true class = which contiguous block branch n falls into (data is
        # sorted into n_classes equal blocks); predicted class = encoding
        # part of the ansatz basis index a (a % enc_dim, the LSBs).
        n_idx = np.arange(N)
        true_class = n_idx // (N // self.n_classes)
        a_idx = np.arange(self.dim)
        pred_class = a_idx % self.enc_dim
        match = true_class[:, None] == pred_class[None, :]
        p1 = float(probs[match].sum())
        return {"0": 1.0 - p1, "1": p1}

    def cost(self, all_data, variational):
        p = self.label_probability(all_data, variational)
        return float(-np.log(max(p.get("1", 0), 1e-12)))
