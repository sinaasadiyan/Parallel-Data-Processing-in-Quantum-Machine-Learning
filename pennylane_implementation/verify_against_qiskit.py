"""
Sanity check: load the SPSA-trained parameters produced by the original Qiskit
notebook (results/<family>/dataset<n>/method{1,2}.json) and re-evaluate them
with the PennyLane circuits in qml_core.py. If the two implementations agree,
predictions/accuracy should match (Method 1 exactly; Method 2 up to the
final-cost / accuracy numbers, since it is read out the same way as Method 1).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from qml_core import Method1Classifier, Method2Classifier  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_dataset(family, num):
    d = os.path.join(ROOT, "datasets", family, f"dataset{num}")
    train_data = pd.read_csv(os.path.join(d, "train_data")).to_numpy()
    train_labels = pd.read_csv(os.path.join(d, "train_labels")).to_numpy().flatten()
    test_data = pd.read_csv(os.path.join(d, "test_data")).to_numpy()
    test_labels = pd.read_csv(os.path.join(d, "test_labels")).to_numpy().flatten()
    return train_data, train_labels, test_data, test_labels


def check_method1(family, num):
    train_data, train_labels, test_data, test_labels = load_dataset(family, num)
    with open(os.path.join(ROOT, "results", family, f"dataset{num}", "method1.json"), encoding="utf-8") as f:
        saved = json.load(f)
    params = np.array(saved["final_results"]["final_parameters"])

    clf = Method1Classifier(n_features=train_data.shape[1], num_reupload=4)
    assert clf.num_parameters == len(params), (clf.num_parameters, len(params))

    test_acc, test_pred = clf.performance_evaluation(test_data, test_labels, params)
    train_acc, train_pred = clf.performance_evaluation(train_data, train_labels, params)

    qiskit_test_pred = saved["final_results"]["test_predictions"]
    qiskit_train_pred = saved["final_results"]["train_predictions"]
    qiskit_test_acc = saved["final_results"]["final_test_accuracy"]
    qiskit_train_acc = saved["final_results"]["final_train_accuracy"]

    match_test = sum(int(a == b) for a, b in zip(test_pred, qiskit_test_pred)) / len(test_pred)
    match_train = sum(int(a == b) for a, b in zip(train_pred, qiskit_train_pred)) / len(train_pred)

    print(f"[Method1] {family}/dataset{num}: "
          f"PennyLane test_acc={test_acc:.4f} (qiskit {qiskit_test_acc:.4f}), "
          f"PennyLane train_acc={train_acc:.4f} (qiskit {qiskit_train_acc:.4f}), "
          f"prediction agreement: test={match_test:.4f} train={match_train:.4f}")
    return match_test, match_train


def check_method2(family, num):
    train_data, train_labels, test_data, test_labels = load_dataset(family, num)
    with open(os.path.join(ROOT, "results", family, f"dataset{num}", "method2.json"), encoding="utf-8") as f:
        saved = json.load(f)
    params = np.array(saved["final_results"]["final_parameters"])

    # Method 2 is trained via the batch circuit, but accuracy/predictions are
    # always read out with the Method-1 per-sample circuit (see README).
    clf1 = Method1Classifier(n_features=train_data.shape[1], num_reupload=4)
    assert clf1.num_parameters == len(params)

    test_acc, test_pred = clf1.performance_evaluation(test_data, test_labels, params)
    train_acc, train_pred = clf1.performance_evaluation(train_data, train_labels, params)

    qiskit_test_pred = saved["final_results"]["test_predictions"]
    qiskit_train_pred = saved["final_results"]["train_predictions"]
    qiskit_test_acc = saved["final_results"]["final_test_accuracy"]
    qiskit_train_acc = saved["final_results"]["final_train_accuracy"]

    match_test = sum(int(a == b) for a, b in zip(test_pred, qiskit_test_pred)) / len(test_pred)
    match_train = sum(int(a == b) for a, b in zip(train_pred, qiskit_train_pred)) / len(train_pred)

    print(f"[Method2 params, Method1 readout] {family}/dataset{num}: "
          f"PennyLane test_acc={test_acc:.4f} (qiskit {qiskit_test_acc:.4f}), "
          f"PennyLane train_acc={train_acc:.4f} (qiskit {qiskit_train_acc:.4f}), "
          f"prediction agreement: test={match_test:.4f} train={match_train:.4f}")
    return match_test, match_train


def check_method2_native_circuit(family, num, n_check=8):
    """Directly exercise the Method-2 batch circuit (independent of Method 1)
    and confirm it reproduces the saved final_cost (= -log P(label=1)) for the
    *training* batch when fed the Method-2-trained parameters."""
    train_data, train_labels, _, _ = load_dataset(family, num)
    with open(os.path.join(ROOT, "results", family, f"dataset{num}", "method2.json"), encoding="utf-8") as f:
        saved = json.load(f)
    params = np.array(saved["final_results"]["final_parameters"])
    qiskit_final_cost = saved["final_results"]["final_cost"]

    clf2 = Method2Classifier(n_features=train_data.shape[1], num_reupload=4, num_data=len(train_data))
    cost = clf2.cost(train_data, params)
    print(f"[Method2 native circuit] {family}/dataset{num}: "
          f"PennyLane cost={cost:.6f}  qiskit final_cost={qiskit_final_cost:.6f}  "
          f"diff={abs(cost - qiskit_final_cost):.2e}")
    return abs(cost - qiskit_final_cost)


if __name__ == "__main__":
    families_nums = [("checkerboard", 1), ("circles", 2), ("corners", 3), ("semicircles", 4)]
    print("=== Method 1 (per-sample circuit) ===")
    for fam, n in families_nums:
        check_method1(fam, n)

    print("\n=== Method 2 trained params, evaluated with Method-1 readout (matches notebook's reported accuracy) ===")
    for fam, n in families_nums:
        check_method2(fam, n)

    print("\n=== Method 2 native batch circuit (cost sanity check, slower) ===")
    for fam, n in families_nums[:1]:
        check_method2_native_circuit(fam, n)
