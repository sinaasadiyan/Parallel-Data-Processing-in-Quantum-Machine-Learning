"""
Core PennyLane re-implementation of the two circuit constructions used in
``Parallel_QML.ipynb`` (originally written with Qiskit).

Method 1 -- one circuit per data point (standard per-sample variational classifier).
Method 2 -- the whole dataset encoded in superposition through an index register
            (QRAM-style batch encoding), with the cost evaluated in a single circuit.

Both methods use:
  - a ZZFeatureMap encoder (Havlicek et al., 2 features, reps=2, full entanglement),
  - a TwoLocal(ry, rz, cz, full, reps=1) variational ansatz, re-uploaded 4 times.

These are reimplemented gate-for-gate (verified against Qiskit's
``ZZFeatureMap``/``TwoLocal`` decompositions) so that parameters trained with the
original Qiskit notebook can be loaded here and reproduce the same predictions.
"""
import itertools
import numpy as np
import pennylane as qml


class OpaqueBlock(qml.operation.Operation):
    """A label-only placeholder operation, used solely to draw compact circuit
    diagrams (it has no decomposition / matrix and must never be executed).
    Mirrors how the original Qiskit notebook bundles each encoder/ansatz block
    into a single named ``Gate`` (via ``.to_gate()``) for its circuit drawings.
    """
    grad_method = None

    def __init__(self, wires, label_text="Block"):
        self._label_text = label_text
        super().__init__(wires=wires)

    def label(self, decimals=None, base_label=None, cache=None):
        return self._label_text

    def decomposition(self):
        return []


def default_device(wires):
    """Prefer the fast C++ ``lightning.qubit`` simulator, fall back to default.qubit."""
    try:
        return qml.device("lightning.qubit", wires=wires)
    except Exception:
        return qml.device("default.qubit", wires=wires)


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------

def zz_feature_map_qfunc(x, wires, reps=2):
    """Second-order Pauli-Z (ZZFeatureMap) encoder, full entanglement.

    ``x`` may be a single sample (1-D, shape ``(n_features,)``) or a batch
    (2-D, shape ``(n_samples, n_features)``) -- ``x[..., i]`` supports both via
    PennyLane's parameter broadcasting, which lets a whole dataset be
    evaluated in a single circuit execution instead of one per sample.
    """
    n = len(wires)
    for _ in range(reps):
        for i in range(n):
            qml.Hadamard(wires=wires[i])
            qml.PhaseShift(2 * x[..., i], wires=wires[i])
        for i, j in itertools.combinations(range(n), 2):
            qml.CNOT(wires=[wires[i], wires[j]])
            qml.PhaseShift(2 * (np.pi - x[..., i]) * (np.pi - x[..., j]), wires=wires[j])
            qml.CNOT(wires=[wires[i], wires[j]])


def zz_feature_map_matrix(x, n_features, reps=2):
    """Dense unitary matrix of the ZZFeatureMap for a fixed data point `x`."""
    wires = list(range(n_features))
    return qml.matrix(zz_feature_map_qfunc, wire_order=wires)(x, wires, reps=reps)


def two_local_num_params(n_features, reps=1):
    """Number of trainable parameters of TwoLocal(['ry','rz'], 'cz', 'full', reps)."""
    return 2 * n_features * (reps + 1)


def two_local_block(params, wires, reps=1):
    """TwoLocal(['ry', 'rz'], 'cz', 'full', reps) variational ansatz."""
    n = len(wires)
    idx = [0]

    def rotation_layer():
        # TwoLocal applies one gate type across all qubits before the next
        # (ry on every qubit, then rz on every qubit) -- not interleaved per qubit.
        for w in wires:
            qml.RY(params[idx[0]], wires=w); idx[0] += 1
        for w in wires:
            qml.RZ(params[idx[0]], wires=w); idx[0] += 1

    rotation_layer()
    for _ in range(reps):
        for i, j in itertools.combinations(range(n), 2):
            qml.CZ(wires=[wires[i], wires[j]])
        rotation_layer()


# --------------------------------------------------------------------------
# Method 1: one circuit per data point
# --------------------------------------------------------------------------

class Method1Classifier:
    """Per-sample variational classifier (re-uploaded ZZFeatureMap + TwoLocal)."""

    def __init__(self, n_features, num_reupload):
        self.n_features = n_features
        self.num_reupload = num_reupload
        self.block_params = two_local_num_params(n_features, reps=1)
        self.num_parameters = num_reupload * self.block_params
        self.wires = list(range(n_features))
        self.dev = qml.device("default.qubit", wires=n_features)
        self._qnode = qml.QNode(self._circuit, self.dev)

    def _circuit(self, data, variational):
        # `data` can be a single sample or a full batch -- see zz_feature_map_qfunc.
        for i in range(self.num_reupload):
            zz_feature_map_qfunc(data, self.wires, reps=2)
            two_local_block(
                variational[i * self.block_params:(i + 1) * self.block_params],
                self.wires, reps=1,
            )
        return qml.probs(wires=[0])

    def label_probability(self, data, variational):
        p0, p1 = self._qnode(np.asarray(data, dtype=float), np.asarray(variational, dtype=float))
        return {"0": float(p0), "1": float(p1)}

    def classification_probability(self, all_data, variational):
        # Single batched circuit execution (parameter broadcasting) instead of
        # one QNode call per sample -- ~100x faster for a 128-point dataset.
        all_data = np.asarray(all_data, dtype=float)
        variational = np.asarray(variational, dtype=float)
        probs = np.atleast_2d(self._qnode(all_data, variational))
        return [{"0": float(p0), "1": float(p1)} for p0, p1 in probs]

    def predict(self, all_data, variational):
        probs = self.classification_probability(all_data, variational)
        return [0 if p.get("0", 0) >= p.get("1", 0) else 1 for p in probs]

    def performance_evaluation(self, data, labels, variational):
        probs = self.classification_probability(data, variational)
        predictions = [0 if p.get("0", 0) >= p.get("1", 0) else 1 for p in probs]
        accuracy = np.mean([int(pr == lb) for pr, lb in zip(predictions, labels)])
        return accuracy, predictions

    def cost(self, all_data, labels, variational):
        probs = self.classification_probability(all_data, variational)
        losses = [-np.log(max(p.get(str(int(lb)), 1e-12), 1e-12))
                   for p, lb in zip(probs, labels)]
        return float(np.mean(losses))


# --------------------------------------------------------------------------
# Method 2: whole dataset in superposition (QRAM-style batch encoding)
# --------------------------------------------------------------------------

class Method2Classifier:
    """
    Batch classifier: all training points are encoded in superposition via an
    index register. Requires ``len(all_data)`` to be a power of two AND the
    data to be ordered with all label-0 points first, all label-1 points last
    (this is how Method 2 implicitly "sees" the labels, exactly mirroring the
    original Qiskit circuit -- see qc2() in Parallel_QML.ipynb).
    """

    def __init__(self, n_features, num_reupload, num_data):
        self.n_features = n_features
        self.num_reupload = num_reupload
        self.num_data = num_data
        self.n_index = int(round(np.log2(num_data)))
        if 2 ** self.n_index != num_data:
            raise ValueError("Method 2 requires the dataset size to be a power of two.")

        self.index_wires = list(range(self.n_index))
        self.feature_wires = list(range(self.n_index, self.n_index + n_features))
        self.label_wire = self.n_index + n_features
        self.total_wires = self.n_index + n_features + 1

        self.block_params = two_local_num_params(n_features, reps=1)
        self.num_parameters = num_reupload * self.block_params

        self.dev = default_device(self.total_wires)
        self._qnode = qml.QNode(self._circuit, self.dev)

    def _circuit(self, matrices, variational):
        for w in self.index_wires:
            qml.Hadamard(wires=w)

        for layer in range(self.num_reupload):
            for i, mat in enumerate(matrices):
                ctrl_vals = [(i >> k) & 1 for k in range(self.n_index)]
                qml.ControlledQubitUnitary(
                    mat,
                    control_wires=self.index_wires,
                    wires=self.feature_wires,
                    control_values=ctrl_vals,
                )
            two_local_block(
                variational[layer * self.block_params:(layer + 1) * self.block_params],
                self.feature_wires, reps=1,
            )

        msb_wire = self.index_wires[-1]
        feat0_wire = self.feature_wires[0]
        qml.ctrl(qml.PauliX, control=[msb_wire, feat0_wire], control_values=[0, 0])(wires=self.label_wire)
        qml.ctrl(qml.PauliX, control=[msb_wire, feat0_wire], control_values=[1, 1])(wires=self.label_wire)

        return qml.probs(wires=[self.label_wire])

    def _matrices(self, all_data):
        return [zz_feature_map_matrix(x, self.n_features, reps=2) for x in all_data]

    def label_probability(self, all_data, variational):
        matrices = self._matrices(all_data)
        p0, p1 = self._qnode(matrices, np.asarray(variational, dtype=float))
        return {"0": float(p0), "1": float(p1)}

    def cost(self, all_data, variational):
        p = self.label_probability(all_data, variational)
        return float(-np.log(max(p.get("1", 0), 1e-12)))
