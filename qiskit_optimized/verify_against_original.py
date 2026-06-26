"""
Sanity check: load the SPSA-trained parameters produced by the original
notebook (../results/<family>/dataset<n>/method{1,2}.json) and re-evaluate
them with the optimized classifiers in qc_core.py. Confirms bit-identical
predictions and an exact cost match -- i.e. the speedup in qc_core.py changes
nothing about what's being computed.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from qc_core import Method1Classifier, Method2Classifier  # noqa: E402

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
    test_acc, test_pred = clf.performance_evaluation(test_data, test_labels, params)
    qiskit_test_pred = saved["final_results"]["test_predictions"]
    agreement = np.mean([int(a == b) for a, b in zip(test_pred, qiskit_test_pred)])

    print(f"[Method1] {family}/dataset{num}: optimized test_acc={test_acc:.4f} "
          f"(original {saved['final_results']['final_test_accuracy']:.4f}), "
          f"prediction agreement={agreement:.4f}")
    return agreement


def check_method2_cost(family, num):
    train_data, _, _, _ = load_dataset(family, num)
    with open(os.path.join(ROOT, "results", family, f"dataset{num}", "method2.json"), encoding="utf-8") as f:
        saved = json.load(f)
    params = np.array(saved["final_results"]["final_parameters"])

    clf2 = Method2Classifier(n_features=train_data.shape[1], num_reupload=4, num_data=len(train_data))
    cost = clf2.cost(train_data, params)
    qiskit_cost = saved["final_results"]["final_cost"]
    print(f"[Method2] {family}/dataset{num}: optimized cost={cost:.6f}  "
          f"original cost={qiskit_cost:.6f}  diff={abs(cost - qiskit_cost):.2e}")
    return abs(cost - qiskit_cost)


if __name__ == "__main__":
    families_nums = [("checkerboard", 1), ("circles", 2), ("corners", 3), ("semicircles", 4)]
    print("=== Method 1 ===")
    for fam, n in families_nums:
        check_method1(fam, n)

    print("\n=== Method 2 (cost match) ===")
    for fam, n in families_nums:
        check_method2_cost(fam, n)
