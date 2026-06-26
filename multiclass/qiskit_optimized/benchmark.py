"""
Benchmarks the optimized multiclass classifiers (qc_core.py) against the
*original* qc1()/qc2() + AerSimulator + transpile() approach from
../Parallel_QML_Checkerboard4L.ipynb, on the same data and parameters, and
confirms the costs match exactly.

Method 1 is benchmarked head-to-head on the real 128-point training set.

Method 2's original approach is only benchmarked on small subsets (4, 8
points) -- the real case (n1=7, 128 points, 4-qubit controlled-ansatz
register) would take a prohibitively long time to even transpile. The
optimized version handles the full 128-point case directly.

Run with: python benchmark.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit.quantum_info import partial_trace
from qiskit_aer import AerSimulator

sys.path.insert(0, os.path.dirname(__file__))
from qc_core import Method1Classifier, Method2Classifier  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUM_FEATURES = 2
NUM_ANSATZ_Q = 2 * NUM_FEATURES
NUM_REUPLOAD = 2


# --------------------------------------------------------------------------
# Original (unoptimized) Method 1, copied verbatim from
# Parallel_QML_Checkerboard4L.ipynb
# --------------------------------------------------------------------------

def original_qc1(data, variational):
    n = NUM_FEATURES
    qreg1 = QuantumRegister(n, "q")
    qreg2 = QuantumRegister(n, "p")
    qc = QuantumCircuit(qreg1, qreg2)
    enc = ZZFeatureMap(feature_dimension=n)
    enc.assign_parameters(dict(zip(enc.parameters, data)), inplace=True)
    enc_gate = enc.to_gate(); enc_gate.name = "Enc"
    for i in range(NUM_REUPLOAD):
        var = TwoLocal(NUM_ANSATZ_Q, ["ry", "rz"], "cz", "full", reps=1)
        values = variational[i * var.num_parameters:(i + 1) * var.num_parameters]
        var.assign_parameters(dict(zip(var.parameters, values)), inplace=True)
        var_gate = var.to_gate(); var_gate.name = f"Var{i + 1}"
        qc.append(enc_gate, qreg1)
        qc.append(var_gate, range(NUM_ANSATZ_Q))
    qc.save_statevector(label="sv")
    return qc


def original_cost_function_1(all_data, labels, variational):
    backend = AerSimulator()
    circs = [original_qc1(d, variational) for d in all_data]
    results = backend.run(transpile(circs, backend=backend)).result()
    losses = []
    for i, lb in enumerate(labels):
        sv = results.data(i)["sv"]
        reduced = partial_trace(sv, list(range(NUM_FEATURES, NUM_ANSATZ_Q)))
        p = reduced.probabilities_dict()
        losses.append(-np.log(p.get(format(int(lb), "02b"), 1e-12)))
    return float(np.mean(losses))


# --------------------------------------------------------------------------
# Original (unoptimized) Method 2's controlled-encoder construction, for the
# small-subset scaling demonstration only.
# --------------------------------------------------------------------------

def original_qc2_subset(all_data, variational, n1):
    n2 = NUM_ANSATZ_Q
    qreg_1 = QuantumRegister(n1, "r")
    qreg_2 = QuantumRegister(n2, "q")
    qreg_3 = QuantumRegister(1, "l")
    qc = QuantumCircuit(qreg_1, qreg_2, qreg_3)
    qc.h(qreg_1)
    enc = QuantumCircuit(n1 + NUM_FEATURES)
    for i in range(len(all_data)):
        fm = ZZFeatureMap(feature_dimension=NUM_FEATURES)
        fm.assign_parameters(dict(zip(fm.parameters, all_data[i])), inplace=True)
        fm_gate = fm.to_gate(label="ZZMap")
        enc.append(fm_gate.control(num_ctrl_qubits=n1, ctrl_state=format(i, f"0{n1}b")),
                   range(n1 + NUM_FEATURES))
    enc_gate = enc.to_gate(); enc_gate.name = "Enc"
    for i in range(NUM_REUPLOAD):
        var = TwoLocal(n2, ["ry", "rz"], "cz", "full", reps=1)
        values = variational[i * 16:(i + 1) * 16]
        var.assign_parameters(dict(zip(var.parameters, values)), inplace=True)
        var_gate = var.to_gate(); var_gate.name = f"Var{i + 1}"
        qc.append(enc_gate, range(n1 + NUM_FEATURES))
        qc.append(var_gate, range(n1, n1 + n2))
    qc.save_statevector(label="sv")
    return qc


def load_dataset(num=1):
    d = os.path.join(ROOT, "datasets", f"dataset{num}")
    train_data = pd.read_csv(os.path.join(d, "train_data")).to_numpy()
    train_labels = pd.read_csv(os.path.join(d, "train_labels")).to_numpy().flatten()
    return train_data, train_labels


def benchmark_method1():
    train_data, train_labels = load_dataset()
    np.random.seed(0)
    variational = np.random.uniform(-np.pi, np.pi, size=32)

    print("=== Method 1: original vs optimized, full 128-point training set ===")

    t0 = time.time()
    original_cost = original_cost_function_1(train_data, train_labels, variational)
    t_original = time.time() - t0

    clf1 = Method1Classifier(n_features=NUM_FEATURES, num_reupload=NUM_REUPLOAD)
    t0 = time.time()
    optimized_cost = clf1.cost(train_data, train_labels, variational)
    t_optimized_cold = time.time() - t0
    t0 = time.time()
    optimized_cost2 = clf1.cost(train_data, train_labels, variational)
    t_optimized_warm = time.time() - t0

    print(f"  original   : {t_original:.3f}s/evaluation   cost={original_cost:.6f}")
    print(f"  optimized  : {t_optimized_cold:.3f}s (1st call, builds cache)   cost={optimized_cost:.6f}")
    print(f"  optimized  : {t_optimized_warm:.3f}s (cache warm, as in every SPSA iteration after the 1st)")
    print(f"  cost match : diff={abs(original_cost - optimized_cost):.2e}")
    print(f"  speedup (warm cache, the relevant number for a 400-iteration training run): "
          f"{t_original / t_optimized_warm:.0f}x\n")


def benchmark_method2_scaling():
    train_data, train_labels = load_dataset()
    np.random.seed(0)
    variational = np.random.uniform(-np.pi, np.pi, size=32)

    print("=== Method 2: original approach's transpiled-circuit-size scaling ===")
    print("    (full n1=7/128-point case is infeasible to transpile in reasonable time;")
    print("     the optimized version handles it directly -- see below)")
    for subset_size, n1 in [(4, 2), (8, 3)]:
        subset = train_data[:subset_size]
        backend = AerSimulator()
        t0 = time.time()
        qc = original_qc2_subset(subset, variational, n1)
        tqc = transpile(qc, backend=backend)
        elapsed = time.time() - t0
        print(f"  n1={n1} ({subset_size:3d} points): build+transpile={elapsed:6.2f}s   "
              f"transpiled gate count={tqc.size():,}")

    print("\n=== Method 2: optimized, full 128-point training set ===")
    clf2 = Method2Classifier(n_features=NUM_FEATURES, num_reupload=NUM_REUPLOAD, num_data=len(train_data))
    t0 = time.time()
    cost = clf2.cost(train_data, variational)
    t_cold = time.time() - t0
    t0 = time.time()
    cost2 = clf2.cost(train_data, variational)
    t_warm = time.time() - t0
    print(f"  optimized  : {t_cold:.3f}s (1st call, builds cache)   cost={cost:.6f}")
    print(f"  optimized  : {t_warm:.3f}s (cache warm, as in every SPSA iteration after the 1st)")
    print(f"  -> at this rate, a full 400-SPSA-iteration training run (3 cost evals/iteration) "
          f"takes roughly {400 * 3 * t_warm:.1f}s\n")


if __name__ == "__main__":
    benchmark_method1()
    benchmark_method2_scaling()
