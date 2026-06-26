"""
Core PennyLane re-implementation of the two circuit constructions used in
``Parallel_QML_Checkerboard4L.ipynb`` (originally written with Qiskit).

This is the multiclass (4-class checkerboard) counterpart of
``../../pennylane_implementation/qml_core.py``. Same overall recipe, but:
  - the variational ansatz acts on 4 qubits (2 encoding qubits + 2 extra
    "workspace" qubits), instead of 2,
  - re-uploaded 2 times instead of 4,
  - the classification readout keeps the 2 *encoding* qubits (not the extra
    workspace qubits!) and interprets their 2-bit joint state as one of 4
    classes -- see the `Method1Classifier` docstring below for why.

Method 1 -- one circuit per data point (standard per-sample variational classifier).
Method 2 -- the whole dataset encoded in superposition through an index register
            (QRAM-style batch encoding), with the cost evaluated in a single circuit.
"""
import itertools
import numpy as np
import pennylane as qml


class OpaqueBlock(qml.operation.Operation):
    """A label-only placeholder operation, used solely to draw compact circuit
    diagrams (it has no decomposition / matrix and must never be executed).
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
# Building blocks (identical to the binary-classification implementation)
# --------------------------------------------------------------------------

def zz_feature_map_qfunc(x, wires, reps=2):
    """Second-order Pauli-Z (ZZFeatureMap) encoder, full entanglement.

    ``x`` may be a single sample (1-D, shape ``(n_features,)``) or a batch
    (2-D, shape ``(n_samples, n_features)``) -- ``x[..., i]`` supports both via
    PennyLane's parameter broadcasting.
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


def two_local_num_params(n_qubits, reps=1):
    """Number of trainable parameters of TwoLocal(['ry','rz'], 'cz', 'full', reps)."""
    return 2 * n_qubits * (reps + 1)


def two_local_block(params, wires, reps=1):
    """TwoLocal(['ry', 'rz'], 'cz', 'full', reps) variational ansatz.

    Verified (see multiclass/pennylane_implementation/verify_against_qiskit.py)
    to reproduce Qiskit's TwoLocal(4, ...) statevector exactly for n=4 qubits:
    Qiskit's circuit interleaves entanglers with rotations differently in its
    gate *list*, but since gates on disjoint qubits commute, the resulting
    unitary is identical to this simpler [rotate][entangle][rotate]... layout,
    as long as the parameter *block* order (all RY across qubits, then all RZ
    across qubits, per rotation layer) and the CZ pair order
    (`itertools.combinations`) match -- which they do.
    """
    n = len(wires)
    idx = [0]

    def rotation_layer():
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
    """Per-sample variational classifier for the 4-class checkerboard task.

    The ansatz acts on ``2 * n_features`` qubits: the first ``n_features``
    ("encoding" qubits, where the ZZFeatureMap is re-applied at every
    re-upload) and ``n_features`` extra "workspace" qubits that only ever see
    the TwoLocal ansatz. Classification reads out the *encoding* qubits'
    joint state as a ``n_features``-bit class label (so 2 qubits -> 4
    classes) -- this exactly mirrors `qc1()`/`label_probability_1()` in the
    original notebook, where `partial_trace` keeps qubits
    ``[0, n_features)`` and discards ``[n_features, 2*n_features)``.
    """

    def __init__(self, n_features, num_reupload):
        self.n_features = n_features
        self.n_ansatz = 2 * n_features
        self.num_reupload = num_reupload
        self.block_params = two_local_num_params(self.n_ansatz, reps=1)
        self.num_parameters = num_reupload * self.block_params
        self.n_classes = 2 ** n_features

        self.encoding_wires = list(range(n_features))
        self.ansatz_wires = list(range(self.n_ansatz))
        # class = sum_k readout_bit_k * 2**k (matches Qiskit's int(bitstring, 2)
        # with qubit 0 as the bitstring's LSB) -> list MSB-qubit first for qml.probs.
        self.readout_wires = list(reversed(self.encoding_wires))

        self.dev = qml.device("default.qubit", wires=self.n_ansatz)
        self._qnode = qml.QNode(self._circuit, self.dev)

    def _circuit(self, data, variational):
        for i in range(self.num_reupload):
            zz_feature_map_qfunc(data, self.encoding_wires, reps=2)
            two_local_block(
                variational[i * self.block_params:(i + 1) * self.block_params],
                self.ansatz_wires, reps=1,
            )
        return qml.probs(wires=self.readout_wires)

    def classification_probability(self, all_data, variational):
        # Single batched circuit execution (parameter broadcasting) instead of
        # one QNode call per sample.
        all_data = np.asarray(all_data, dtype=float)
        variational = np.asarray(variational, dtype=float)
        probs = np.atleast_2d(self._qnode(all_data, variational))
        return probs  # shape (n_samples, n_classes)

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
        losses = [-np.log(max(p[int(lb)], 1e-12)) for p, lb in zip(probs, labels)]
        return float(np.mean(losses))


# --------------------------------------------------------------------------
# Method 2: whole dataset in superposition (QRAM-style batch encoding)
# --------------------------------------------------------------------------

class Method2Classifier:
    """
    Batch classifier: all training points are encoded in superposition via an
    index register. Requires ``len(all_data)`` to be a power of two AND the
    data to be ordered in ``n_classes`` contiguous, equally-sized, increasing
    blocks (all class-0 points first, ..., all class-(n_classes-1) points
    last). The *top* ``n_features`` bits of the index register then equal the
    true class, which is how Method 2 implicitly "sees" the labels -- exactly
    mirroring `qc2()` in the original Qiskit notebook.
    """

    def __init__(self, n_features, num_reupload, num_data):
        self.n_features = n_features
        self.n_ansatz = 2 * n_features
        self.num_reupload = num_reupload
        self.num_data = num_data
        self.n_classes = 2 ** n_features

        self.n_index = int(round(np.log2(num_data)))
        if 2 ** self.n_index != num_data:
            raise ValueError("Method 2 requires the dataset size to be a power of two.")
        if num_data % self.n_classes != 0:
            raise ValueError("Method 2 requires the dataset size to be divisible by n_classes.")

        self.index_wires = list(range(self.n_index))
        self.ansatz_wires = list(range(self.n_index, self.n_index + self.n_ansatz))
        self.encoding_wires = self.ansatz_wires[:n_features]  # first n_features ansatz qubits
        self.label_wire = self.n_index + self.n_ansatz
        self.total_wires = self.n_index + self.n_ansatz + 1

        self.block_params = two_local_num_params(self.n_ansatz, reps=1)
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
                    wires=self.encoding_wires,
                    control_values=ctrl_vals,
                )
            two_local_block(
                variational[layer * self.block_params:(layer + 1) * self.block_params],
                self.ansatz_wires, reps=1,
            )

        # Flip the label qubit whenever (true class, encoded in the top
        # n_features bits of the index register) == (predicted class, read
        # from the encoding qubits): one multi-controlled-X per class.
        index_top_wires = [self.index_wires[-1 - k] for k in range(self.n_features)]  # MSB..LSB of true class
        pred_wires = self.encoding_wires  # q0 (LSB) .. q_{n_features-1} (MSB) of predicted class
        for c in range(self.n_classes):
            true_bits = [(c >> (self.n_features - 1 - k)) & 1 for k in range(self.n_features)]
            pred_bits = [(c >> k) & 1 for k in range(self.n_features)]
            qml.ctrl(
                qml.PauliX,
                control=index_top_wires + pred_wires,
                control_values=true_bits + pred_bits,
            )(wires=self.label_wire)

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
