#!/usr/bin/env python3
"""
🚀 Automatic QML Results Analyzer (with SEM)
📊 Comprehensive analysis of all SPSA training results
➕ Includes Standard Error of the Mean (SEM) alongside averages in the report

Usage: python auto_results_analyzer_sem.py

Notes:
- This script is a SEM-augmented variant. It does NOT modify the original analyzer.
- It adds ± SEM to reported averages (loss, train acc, test acc).

Author: QML Analysis System
Version: 2.1 (SEM)
"""

import json
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import pandas as pd
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class QMLResultsAnalyzerSEM:
    """Comprehensive QML Results Analysis System (with SEM in report)"""

    def __init__(self, results_base_dir="results", output_dir="analysis_plots"):
        self.results_base_dir = Path(results_base_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Analysis timestamp
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def log(self, message, level="INFO"):
        """Enhanced logging with emojis"""
        emoji_map = {
            "INFO": "ℹ️",
            "SUCCESS": "✅",
            "WARNING": "⚠️",
            "ERROR": "❌",
            "SEARCH": "🔍",
            "PLOT": "📊",
            "SAVE": "💾",
        }
        print(f"{emoji_map.get(level, 'ℹ️')} {message}")

    def correct_accuracy(self, accuracy):
        """Apply accuracy correction: if < 0.5, subtract from 1"""
        if accuracy < 0.5:
            return 1.0 - accuracy
        return accuracy

    def extract_experiment_info(self, folder_name):
        """Extract experiment information from folder name"""
        # Support multiple naming patterns
        patterns = [
            r'checkerboard-4L-4Q-dataset(\d+)',  # Original pattern
            r'dataset(\d+)_run(\d+)',  # Alternative pattern
            r'spsa_loss(\d+)_dataset(\d+)_run(\d+)',  # New loss pattern
            r'exp(\d+)_dataset(\d+)_run(\d+)',  # Experiment pattern
        ]

        for pattern in patterns:
            match = re.match(pattern, folder_name)
            if match:
                groups = match.groups()
                if len(groups) == 1:
                    return {
                        'dataset': int(groups[0]),
                        'experiment_type': 'standard',
                    }
                elif len(groups) == 2:
                    return {
                        'loss_version': int(groups[0]),
                        'dataset': int(groups[1]),
                        'experiment_type': 'loss_variant',
                    }

        return None

    def find_all_result_folders(self):
        """Discover all result folders in the results directory"""
        if not self.results_base_dir.exists():
            self.log(f"Results directory not found: {self.results_base_dir}", "ERROR")
            return []

        all_folders = []

        # Check root results directory
        for item in self.results_base_dir.iterdir():
            if item.is_dir():
                exp_info = self.extract_experiment_info(item.name)
                if exp_info:
                    all_folders.append({
                        'path': item,
                        'name': item.name,
                        'info': exp_info,
                    })

        # Check subdirectories (like spsa_loss3/)
        for subdir in self.results_base_dir.iterdir():
            if subdir.is_dir() and subdir.name.startswith(('spsa_', 'loss', 'exp')):
                for item in subdir.iterdir():
                    if item.is_dir():
                        exp_info = self.extract_experiment_info(item.name)
                        if exp_info:
                            exp_info['parent_folder'] = subdir.name
                            all_folders.append({
                                'path': item,
                                'name': f"{subdir.name}/{item.name}",
                                'info': exp_info,
                            })

        return all_folders

    def get_latest_results(self, result_dir):
        """Get the latest results from a result directory based on timestamp"""
        json_files = list(result_dir.glob("results_spsa_method*_*.json"))

        if not json_files:
            return None, None

        # Group by method and get latest timestamp for each
        method1_files = [f for f in json_files if 'method1' in f.name]
        method2_files = [f for f in json_files if 'method2' in f.name]

        def get_latest_file(files):
            if not files:
                return None
            # Sort by timestamp in filename
            try:
                latest = max(
                    files,
                    key=lambda f: f.name.split('_')[-2] + f.name.split('_')[-1].split('.')[0],
                )
                return latest
            except Exception:
                return files[0] if files else None

        method1_latest = get_latest_file(method1_files)
        method2_latest = get_latest_file(method2_files)

        return method1_latest, method2_latest

    def load_result_data(self, json_file):
        """Load and extract relevant data from a JSON result file"""
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            final_results = data['final_results']
            return {
                'final_cost': final_results['final_cost'],
                'final_train_accuracy': self.correct_accuracy(final_results['final_train_accuracy']),
                'final_test_accuracy': self.correct_accuracy(final_results['final_test_accuracy']),
                'dataset': data['experiment_info']['dataset'],
                'timestamp': data['experiment_info']['timestamp'],
            }
        except Exception as e:
            self.log(f"Error loading {json_file}: {e}", "ERROR")
            return None

    def collect_all_results(self):
        """Collect all results from discovered folders"""
        self.log("Starting comprehensive results collection...", "SEARCH")

        result_folders = self.find_all_result_folders()

        if not result_folders:
            self.log("No result folders found!", "ERROR")
            return []

        self.log(f"Found {len(result_folders)} result folders", "SUCCESS")

        all_results = []
        processed_folders = 0

        for folder_info in result_folders:
            folder_path = folder_info['path']
            folder_name = folder_info['name']
            exp_info = folder_info['info']

            method1_file, method2_file = self.get_latest_results(folder_path)

            if method1_file:
                method1_data = self.load_result_data(method1_file)
                if method1_data:
                    result_entry = {
                        'dataset': exp_info['dataset'],
                        'method': 'Method 1',
                        'cost': method1_data['final_cost'],
                        'train_accuracy': method1_data['final_train_accuracy'],
                        'test_accuracy': method1_data['final_test_accuracy'],
                        'folder': folder_name,
                        'experiment_type': exp_info['experiment_type'],
                    }

                    # Add additional info for loss variants
                    if 'loss_version' in exp_info:
                        result_entry['loss_version'] = exp_info['loss_version']
                    if 'parent_folder' in exp_info:
                        result_entry['parent_folder'] = exp_info['parent_folder']

                    all_results.append(result_entry)

            if method2_file:
                method2_data = self.load_result_data(method2_file)
                if method2_data:
                    result_entry = {
                        'dataset': exp_info['dataset'],
                        'method': 'Method 2',
                        'cost': method2_data['final_cost'],
                        'train_accuracy': method2_data['final_train_accuracy'],
                        'test_accuracy': method2_data['final_test_accuracy'],
                        'folder': folder_name,
                        'experiment_type': exp_info['experiment_type'],
                    }

                    # Add additional info for loss variants
                    if 'loss_version' in exp_info:
                        result_entry['loss_version'] = exp_info['loss_version']
                    if 'parent_folder' in exp_info:
                        result_entry['parent_folder'] = exp_info['parent_folder']

                    all_results.append(result_entry)

            if method1_file or method2_file:
                processed_folders += 1

        self.log(f"Processed {processed_folders} folders with results", "SUCCESS")
        return all_results

    def _compute_sem(self, values):
        """Compute Standard Error of the Mean (SEM) for a list/array of values."""
        arr = np.array(values, dtype=float)
        n = arr.size
        if n <= 1:
            return 0.0
        return float(np.std(arr, ddof=1) / np.sqrt(n))

    def create_comprehensive_plots(self, all_results):
        """Create comprehensive analysis plots (plots unchanged); compute SEM for report."""

        if not all_results:
            self.log("No results to plot!", "ERROR")
            return

        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(all_results)

        self.log(f"Creating plots for {len(all_results)} results", "PLOT")
        self.log(f"   📊 Method 1: {len(df[df['method'] == 'Method 1'])} results")
        self.log(f"   📊 Method 2: {len(df[df['method'] == 'Method 2'])} results")

        # Get unique datasets
        datasets = sorted(df['dataset'].unique())

        # Prepare data for plotting
        datasets_with_data = []
        method1_min_loss = []
        method2_min_loss = []
        method1_max_train_acc = []
        method2_max_train_acc = []
        method1_max_test_acc = []
        method2_max_test_acc = []

        for dataset in datasets:
            dataset_data = df[df['dataset'] == dataset]

            method1_data = dataset_data[dataset_data['method'] == 'Method 1']
            method2_data = dataset_data[dataset_data['method'] == 'Method 2']

            if len(method1_data) > 0 and len(method2_data) > 0:
                datasets_with_data.append(dataset)
                method1_min_loss.append(method1_data['cost'].min())
                method2_min_loss.append(method2_data['cost'].min())
                method1_max_train_acc.append(method1_data['train_accuracy'].max())
                method2_max_train_acc.append(method2_data['train_accuracy'].max())
                method1_max_test_acc.append(method1_data['test_accuracy'].max())
                method2_max_test_acc.append(method2_data['test_accuracy'].max())

        # Create figure with improved layout: 2 rows × 3 columns
        fig = plt.figure(figsize=(20, 14))

        x_pos = np.arange(len(datasets_with_data))
        width = 0.35

        # Row 1: Train Accuracy (left), Test Accuracy (middle), Method Wins (right)

        # Plot 1: Train Accuracy Comparison
        ax1 = plt.subplot(2, 3, 1)
        bars1 = ax1.bar(
            x_pos - width / 2, method1_max_train_acc, width, label='Method 1', alpha=0.8, color='skyblue'
        )
        bars2 = ax1.bar(
            x_pos + width / 2, method2_max_train_acc, width, label='Method 2', alpha=0.8, color='lightcoral'
        )

        ax1.set_xlabel('Dataset')
        ax1.set_ylabel('Maximum Train Accuracy')
        ax1.set_title('Maximum Train Accuracy per Dataset', fontsize=14, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f'D{d:02d}' for d in datasets_with_data], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2.0, height + 0.01, f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2.0, height + 0.01, f'{height:.3f}', ha='center', va='bottom', fontsize=8)

        # Plot 2: Test Accuracy Comparison
        ax2 = plt.subplot(2, 3, 2)
        bars3 = ax2.bar(
            x_pos - width / 2, method1_max_test_acc, width, label='Method 1', alpha=0.8, color='skyblue'
        )
        bars4 = ax2.bar(
            x_pos + width / 2, method2_max_test_acc, width, label='Method 2', alpha=0.8, color='lightcoral'
        )

        ax2.set_xlabel('Dataset')
        ax2.set_ylabel('Maximum Test Accuracy')
        ax2.set_title('Maximum Test Accuracy per Dataset', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f'D{d:02d}' for d in datasets_with_data], rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])

        # Add value labels on bars
        for bar in bars3:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2.0, height + 0.01, f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        for bar in bars4:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2.0, height + 0.01, f'{height:.3f}', ha='center', va='bottom', fontsize=8)

        # Plot 3: Method Wins Comparison
        ax3 = plt.subplot(2, 3, 3)

        method1_wins_loss = sum(1 for i in range(len(datasets_with_data)) if method1_min_loss[i] < method2_min_loss[i])
        method2_wins_loss = len(datasets_with_data) - method1_wins_loss
        method1_wins_train_acc = sum(
            1 for i in range(len(datasets_with_data)) if method1_max_train_acc[i] > method2_max_train_acc[i]
        )
        method2_wins_train_acc = len(datasets_with_data) - method1_wins_train_acc
        method1_wins_test_acc = sum(
            1 for i in range(len(datasets_with_data)) if method1_max_test_acc[i] > method2_max_test_acc[i]
        )
        method2_wins_test_acc = len(datasets_with_data) - method1_wins_test_acc

        categories = ['Min\nLoss', 'Train\nAcc', 'Test\nAcc']
        method1_wins = [method1_wins_loss, method1_wins_train_acc, method1_wins_test_acc]
        method2_wins = [method2_wins_loss, method2_wins_train_acc, method2_wins_test_acc]

        x_cat = np.arange(len(categories))
        bars5 = ax3.bar(x_cat - width / 2, method1_wins, width, label='Method 1', alpha=0.8, color='skyblue')
        bars6 = ax3.bar(x_cat + width / 2, method2_wins, width, label='Method 2', alpha=0.8, color='lightcoral')

        ax3.set_xlabel('Competition Type')
        ax3.set_ylabel('Number of Dataset Wins')
        ax3.set_title('Method Competition Summary', fontsize=14, fontweight='bold')
        ax3.set_xticks(x_cat)
        ax3.set_xticklabels(categories)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Add value labels
        for bars in [bars5, bars6]:
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width() / 2.0, height + 0.1, f'{int(height)}', ha='center', va='bottom', fontsize=12, fontweight='bold')

        # Row 2: Loss Comparison (left), Scatter Plot (middle), Average Performance (right)

        # Plot 4: Minimum Loss Comparison
        ax4 = plt.subplot(2, 3, 4)
        bars7 = ax4.bar(x_pos - width / 2, method1_min_loss, width, label='Method 1', alpha=0.8, color='skyblue')
        bars8 = ax4.bar(x_pos + width / 2, method2_min_loss, width, label='Method 2', alpha=0.8, color='lightcoral')

        ax4.set_xlabel('Dataset')
        ax4.set_ylabel('Minimum Cost (Loss)')
        ax4.set_title('Minimum Loss per Dataset', fontsize=14, fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels([f'D{d:02d}' for d in datasets_with_data], rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar in bars7:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2.0, height + 0.01, f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        for bar in bars8:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2.0, height + 0.01, f'{height:.3f}', ha='center', va='bottom', fontsize=8)

        # Plot 5: Scatter Plot - Cost vs Test Accuracy
        ax5 = plt.subplot(2, 3, 5)

        method1_df = df[df['method'] == 'Method 1']
        method2_df = df[df['method'] == 'Method 2']

        ax5.scatter(method1_df['cost'], method1_df['test_accuracy'], alpha=0.7, s=80, color='skyblue', label='Method 1')
        ax5.scatter(method2_df['cost'], method2_df['test_accuracy'], alpha=0.7, s=80, color='lightcoral', label='Method 2')

        ax5.set_xlabel('Cost (Loss)')
        ax5.set_ylabel('Test Accuracy')
        ax5.set_title('Cost vs Test Accuracy - All Runs', fontsize=14, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # Plot 6: Average Performance Comparison
        ax6 = plt.subplot(2, 3, 6)

        # Calculate averages
        method1_avg_loss = float(np.mean(method1_min_loss)) if len(method1_min_loss) else float('nan')
        method2_avg_loss = float(np.mean(method2_min_loss)) if len(method2_min_loss) else float('nan')
        method1_avg_train_acc = float(np.mean(method1_max_train_acc)) if len(method1_max_train_acc) else float('nan')
        method2_avg_train_acc = float(np.mean(method2_max_train_acc)) if len(method2_max_train_acc) else float('nan')
        method1_avg_test_acc = float(np.mean(method1_max_test_acc)) if len(method1_max_test_acc) else float('nan')
        method2_avg_test_acc = float(np.mean(method2_max_test_acc)) if len(method2_max_test_acc) else float('nan')

        # Compute SEMs for report
        method1_sem_loss = self._compute_sem(method1_min_loss)
        method2_sem_loss = self._compute_sem(method2_min_loss)
        method1_sem_train_acc = self._compute_sem(method1_max_train_acc)
        method2_sem_train_acc = self._compute_sem(method2_max_train_acc)
        method1_sem_test_acc = self._compute_sem(method1_max_test_acc)
        method2_sem_test_acc = self._compute_sem(method2_max_test_acc)

        # Create grouped bar chart (values only; no SEM error bars requested for plots)
        metrics = ['Avg Min\nLoss', 'Avg Train\nAcc', 'Avg Test\nAcc']
        method1_values = [method1_avg_loss, method1_avg_train_acc, method1_avg_test_acc]
        method2_values = [method2_avg_loss, method2_avg_train_acc, method2_avg_test_acc]

        x_metrics = np.arange(len(metrics))
        bars9 = ax6.bar(x_metrics - width / 2, method1_values, width, label='Method 1', alpha=0.8, color='skyblue')
        bars10 = ax6.bar(x_metrics + width / 2, method2_values, width, label='Method 2', alpha=0.8, color='lightcoral')

        ax6.set_xlabel('Performance Metrics')
        ax6.set_ylabel('Average Values')
        ax6.set_title('Average Performance Comparison', fontsize=14, fontweight='bold')
        ax6.set_xticks(x_metrics)
        ax6.set_xticklabels(metrics)
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        # Add value labels
        for i, (bar1, bar2) in enumerate(zip(bars9, bars10)):
            val1 = method1_values[i]
            val2 = method2_values[i]

            if i == 0:  # Loss values (smaller is better)
                ax6.text(bar1.get_x() + bar1.get_width() / 2.0, bar1.get_height() + 0.02, f'{val1:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
                ax6.text(bar2.get_x() + bar2.get_width() / 2.0, bar2.get_height() + 0.02, f'{val2:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            else:  # Accuracy values
                ax6.text(bar1.get_x() + bar1.get_width() / 2.0, bar1.get_height() + 0.01, f'{val1:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
                ax6.text(bar2.get_x() + bar2.get_width() / 2.0, bar2.get_height() + 0.01, f'{val2:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Adjust layout
        plt.tight_layout(pad=3.0)

        # Add main title with timestamp
        fig.suptitle(
            f'QML SPSA Comprehensive Analysis - {self.timestamp}',
            fontsize=18,
            fontweight='bold',
            y=0.98,
        )

        # Save plot
        output_file = self.output_dir / f'auto_analysis_{self.timestamp}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        self.log(f"Plot saved: {output_file}", "SAVE")

        return {
            'total_results': len(all_results),
            'datasets_analyzed': len(datasets_with_data),
            'method1_wins': {
                'loss': method1_wins_loss,
                'train_acc': method1_wins_train_acc,
                'test_acc': method1_wins_test_acc,
            },
            'method2_wins': {
                'loss': method2_wins_loss,
                'train_acc': method2_wins_train_acc,
                'test_acc': method2_wins_test_acc,
            },
            'averages': {
                'method1': {
                    'loss': method1_avg_loss,
                    'train_acc': method1_avg_train_acc,
                    'test_acc': method1_avg_test_acc,
                },
                'method2': {
                    'loss': method2_avg_loss,
                    'train_acc': method2_avg_train_acc,
                    'test_acc': method2_avg_test_acc,
                },
            },
            'averages_sem': {
                'method1': {
                    'loss': method1_sem_loss,
                    'train_acc': method1_sem_train_acc,
                    'test_acc': method1_sem_test_acc,
                },
                'method2': {
                    'loss': method2_sem_loss,
                    'train_acc': method2_sem_train_acc,
                    'test_acc': method2_sem_test_acc,
                },
            },
        }

    def create_summary_report(self, stats, all_results):
        """Create detailed summary report including ± SEM for averages"""

        # Group results by experiment type and folder
        df = pd.DataFrame(all_results)

        report_lines = [
            f"🚀 QML SPSA Automatic Analysis Report (with SEM)",
            f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"{"=" * 60}",
            f"",
            f"📊 SUMMARY STATISTICS:",
            f"   🔢 Total Results: {stats['total_results']}",
            f"   📈 Datasets Analyzed: {stats['datasets_analyzed']}",
            f"",
            f"🏆 METHOD COMPETITION RESULTS:",
            f"   🔵 Method 1 Wins:",
            f"      • Min Loss: {stats['method1_wins']['loss']}/{stats['datasets_analyzed']}",
            f"      • Train Accuracy: {stats['method1_wins']['train_acc']}/{stats['datasets_analyzed']}",
            f"      • Test Accuracy: {stats['method1_wins']['test_acc']}/{stats['datasets_analyzed']}",
            f"   🔴 Method 2 Wins:",
            f"      • Min Loss: {stats['method2_wins']['loss']}/{stats['datasets_analyzed']}",
            f"      • Train Accuracy: {stats['method2_wins']['train_acc']}/{stats['datasets_analyzed']}",
            f"      • Test Accuracy: {stats['method2_wins']['test_acc']}/{stats['datasets_analyzed']}",
            f"",
            f"📈 AVERAGE PERFORMANCE (± SEM):",
            f"   🔵 Method 1:",
            f"      • Average Min Loss: {stats['averages']['method1']['loss']:.4f} ± {stats['averages_sem']['method1']['loss']:.4f}",
            f"      • Average Train Accuracy: {stats['averages']['method1']['train_acc']:.4f} ± {stats['averages_sem']['method1']['train_acc']:.4f}",
            f"      • Average Test Accuracy: {stats['averages']['method1']['test_acc']:.4f} ± {stats['averages_sem']['method1']['test_acc']:.4f}",
            f"   🔴 Method 2:",
            f"      • Average Min Loss: {stats['averages']['method2']['loss']:.4f} ± {stats['averages_sem']['method2']['loss']:.4f}",
            f"      • Average Train Accuracy: {stats['averages']['method2']['train_acc']:.4f} ± {stats['averages_sem']['method2']['train_acc']:.4f}",
            f"      • Average Test Accuracy: {stats['averages']['method2']['test_acc']:.4f} ± {stats['averages_sem']['method2']['test_acc']:.4f}",
            f"",
        ]

        # Add experiment breakdown
        if 'experiment_type' in df.columns:
            exp_types = df['experiment_type'].value_counts()
            report_lines.extend([
                f"🧪 EXPERIMENT BREAKDOWN:",
                f"   📁 Experiment Types Found:",
            ])
            for exp_type, count in exp_types.items():
                report_lines.append(f"      • {exp_type}: {count} results")
            report_lines.append("")

        # Add folder breakdown
        if 'parent_folder' in df.columns:
            parent_folders = df['parent_folder'].value_counts()
            report_lines.extend([
                f"📂 FOLDER BREAKDOWN:",
                f"   📁 Result Sources:",
            ])
            for folder, count in parent_folders.items():
                report_lines.append(f"      • {folder}: {count} results")
            report_lines.append("")

        # Dataset-wise breakdown
        datasets = sorted(df['dataset'].unique())
        report_lines.extend([
            f"📊 DATASET-WISE RESULTS:",
            f"   📈 Per Dataset Summary:",
        ])

        for dataset in datasets:
            dataset_data = df[df['dataset'] == dataset]
            method1_data = dataset_data[dataset_data['method'] == 'Method 1']
            method2_data = dataset_data[dataset_data['method'] == 'Method 2']

            if len(method1_data) > 0 and len(method2_data) > 0:
                m1_best_loss = method1_data['cost'].min()
                m2_best_loss = method2_data['cost'].min()
                m1_best_test = method1_data['test_accuracy'].max()
                m2_best_test = method2_data['test_accuracy'].max()

                report_lines.extend([
                    f"      📍 Dataset {dataset:02d}:",
                    f"         • Method 1: Loss={m1_best_loss:.4f}, Test Acc={m1_best_test:.4f}",
                    f"         • Method 2: Loss={m2_best_loss:.4f}, Test Acc={m2_best_test:.4f}",
                ])

        report_lines.extend([
            f"",
            f"{"=" * 60}",
            f"🎯 ANALYSIS COMPLETE - Check the generated plot for visual insights!",
            f"📊 Plot file: auto_analysis_{self.timestamp}.png",
        ])

        # Save report
        report_file = self.output_dir / f'auto_analysis_report_{self.timestamp}.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        self.log(f"Report saved: {report_file}", "SAVE")

        return report_lines

    def run_analysis(self):
        """Run complete automatic analysis (with SEM in report)"""

        self.log("🚀 Starting Automatic QML Results Analysis (with SEM)", "INFO")
        self.log("=" * 50)

        # Collect all results
        all_results = self.collect_all_results()

        if not all_results:
            self.log("No results found to analyze!", "ERROR")
            return False

        # Create comprehensive plots and compute stats (including SEM)
        stats = self.create_comprehensive_plots(all_results)

        # Create summary report
        report_lines = self.create_summary_report(stats, all_results)

        # Print summary
        self.log("=" * 50)
        self.log("🎉 Analysis Complete!", "SUCCESS")
        self.log(f"📁 Output Directory: {self.output_dir}")
        self.log(f"📊 Results Plot: auto_analysis_{self.timestamp}.png")
        self.log(f"📝 Summary Report: auto_analysis_report_{self.timestamp}.txt")

        # Quick summary (unchanged)
        self.log("📈 Quick Summary:")
        self.log(f"   🔢 Total Results: {stats['total_results']}")
        self.log(f"   📊 Datasets: {stats['datasets_analyzed']}")
        self.log(f"   🏆 Method 1 Wins: {stats['method1_wins']['test_acc']}/{stats['datasets_analyzed']} (Test Acc)")
        self.log(f"   🏆 Method 2 Wins: {stats['method2_wins']['test_acc']}/{stats['datasets_analyzed']} (Test Acc)")

        return True


def main():
    """Main function"""
    print("🔬 QML Results Automatic Analyzer v2.1 (SEM)")
    print("🚀 Comprehensive analysis of all SPSA training results (± SEM in report)")
    print("=" * 60)

    # Initialize analyzer
    analyzer = QMLResultsAnalyzerSEM()

    # Run analysis
    success = analyzer.run_analysis()

    if success:
        print("\n✅ Analysis completed successfully!")
        print("🔄 You can run this script anytime to get updated results!")
    else:
        print("\n❌ Analysis failed. Check the logs above.")

    return success


if __name__ == "__main__":
    main()


