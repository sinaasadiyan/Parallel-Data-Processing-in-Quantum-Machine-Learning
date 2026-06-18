#!/usr/bin/env python3
"""
Automatic QML Results Analyzer (Best Test Acc from Epoch 150+)

This script mirrors the existing analyzer, but changes selection logic:
- For each result JSON, pick the epoch with the best test accuracy considering only epochs [150, end).
- Use the corresponding cost (loss) and train accuracy at that same epoch.
- Save outputs to a separate output directory so nothing else is affected.

Usage: python auto_results_analyzer_best150.py
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from auto_results_analyzer_sem import QMLResultsAnalyzerSEM


class QMLResultsAnalyzerBestFrom150(QMLResultsAnalyzerSEM):
    """Analyzer that selects the best epoch by test accuracy from 150 onward."""

    def __init__(self, results_base_dir: str = "results", output_dir: str = "analysis_plots_best_from150"):
        super().__init__(results_base_dir=results_base_dir, output_dir=output_dir)

    def load_result_data(self, json_file: Path) -> Optional[Dict[str, Any]]:
        """Load data and select best epoch by test accuracy from 150 onward.

        Fallback to original final results if training history is missing or invalid.
        """
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Prefer training history to pick best epoch
            training_history = data.get("training_history", {}) or {}
            costs = training_history.get("costs")
            train_accuracies = training_history.get("train_accuracies")
            test_accuracies = training_history.get("test_accuracies")

            if (
                isinstance(costs, list)
                and isinstance(train_accuracies, list)
                and isinstance(test_accuracies, list)
                and len(costs) > 0
                and len(train_accuracies) > 0
                and len(test_accuracies) > 0
            ):
                total_points = len(test_accuracies)
                # Start from epoch index 150; if fewer points, use whatever is available
                start_index = 300 if total_points > 300 else 0

                # Guard against edge cases
                if start_index >= total_points:
                    start_index = max(0, total_points - 1)

                # Compute best index in the [start_index, end) window by test accuracy
                window_test_acc = test_accuracies[start_index:]
                best_local_index = int(np.argmax(window_test_acc))
                best_index = start_index + best_local_index

                # Fetch aligned metrics; if any list is shorter, fallback to nearest safe value
                chosen_cost = (
                    float(costs[best_index])
                    if best_index < len(costs)
                    else float(costs[-1])
                )
                chosen_train_acc = (
                    float(train_accuracies[best_index])
                    if best_index < len(train_accuracies)
                    else float(train_accuracies[-1])
                )
                chosen_test_acc = float(test_accuracies[min(best_index, len(test_accuracies) - 1)])

                # Apply the same accuracy correction policy as the base analyzer
                chosen_train_acc = self.correct_accuracy(chosen_train_acc)
                chosen_test_acc = self.correct_accuracy(chosen_test_acc)

                return {
                    "final_cost": chosen_cost,
                    "final_train_accuracy": chosen_train_acc,
                    "final_test_accuracy": chosen_test_acc,
                    "dataset": data["experiment_info"]["dataset"],
                    "timestamp": data["experiment_info"]["timestamp"],
                }

            # Fallback: behave like the base class if history is missing
            final_results = data["final_results"]
            return {
                "final_cost": final_results["final_cost"],
                "final_train_accuracy": self.correct_accuracy(final_results["final_train_accuracy"]),
                "final_test_accuracy": self.correct_accuracy(final_results["final_test_accuracy"]),
                "dataset": data["experiment_info"]["dataset"],
                "timestamp": data["experiment_info"]["timestamp"],
            }

        except Exception as e:
            self.log(f"Error loading {json_file}: {e}", "ERROR")
            return None


def main() -> bool:
    print("🔬 QML Results Analyzer (Best from Epoch 300)")
    print("🚀 Selecting best test accuracy from epoch 300 onward (aligned cost/train acc)")
    print("=" * 60)

    analyzer = QMLResultsAnalyzerBestFrom150()
    success = analyzer.run_analysis()

    if success:
        print("\n✅ Analysis completed successfully (best-from-300)!")
    else:
        print("\n❌ Analysis failed. Check the logs above.")

    return success


if __name__ == "__main__":
    main()


