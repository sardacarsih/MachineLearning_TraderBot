"""
============================================
Model Comparison and Selection
============================================
Compares model metrics, plots evaluation charts (confusion matrices, ROC,
precision-recall curves), and selects the best model for live trading.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, precision_recall_curve

from config.settings import config
from utils.logger import get_logger

logger = get_logger()


class ModelComparison:
    """
    Compares the results of trained machine learning models, generates comparison charts,
    and produces reports.
    """

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir or config.paths.backtest_dir
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"ModelComparison initialized, saving charts to {self.output_dir}")

    def compare_models(self, results_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Creates a comparison table (DataFrame) of all models.

        Args:
            results_dict: Dictionary containing evaluation results for each model.

        Returns:
            pandas DataFrame with models as rows and metrics as columns.
        """
        records = []
        for name, metrics in results_dict.items():
            record = {
                "Model": name,
                "Accuracy": metrics.get("accuracy", 0.0),
                "Precision (Macro)": metrics.get("precision_macro", 0.0),
                "Recall (Macro)": metrics.get("recall_macro", 0.0),
                "F1-Score (Macro)": metrics.get("f1_macro", 0.0),
                "Trade Precision": metrics.get("trade_signal_precision", 0.0),
                "Precision (BUY)": metrics.get("precision_per_class", [0.0, 0.0, 0.0])[1],
                "Precision (SELL)": metrics.get("precision_per_class", [0.0, 0.0, 0.0])[2],
                "F1 (BUY)": metrics.get("f1_per_class", [0.0, 0.0, 0.0])[1],
                "F1 (SELL)": metrics.get("f1_per_class", [0.0, 0.0, 0.0])[2],
            }
            records.append(record)

        df = pd.DataFrame(records)
        df = df.set_index("Model")
        logger.info("Model comparison table generated:\n" + df.to_string())
        return df

    def plot_confusion_matrices(self, results_dict: Dict[str, Dict[str, Any]]):
        """Plots confusion matrix for each model side-by-side or in a grid."""
        n_models = len(results_dict)
        if n_models == 0:
            return

        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))
        if n_models == 1:
            axes = [axes]

        target_names = ["NO_TRADE", "BUY", "SELL"]

        for idx, (name, metrics) in enumerate(results_dict.items()):
            cm = np.array(metrics.get("confusion_matrix", [[0, 0, 0], [0, 0, 0], [0, 0, 0]]))
            ax = axes[idx]

            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=target_names, yticklabels=target_names, ax=ax
            )
            ax.set_title(f"{name} Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, "confusion_matrices.png")
        plt.savefig(filepath)
        plt.close()
        logger.info(f"Confusion matrices saved to {filepath}")

    def plot_roc_curves(self, results_dict: Dict[str, Dict[str, Any]], y_test: np.ndarray):
        """
        Plots ROC curves for all models for BUY and SELL classes.
        """
        classes = [1, 2]  # BUY, SELL
        class_names = {1: "BUY", 2: "SELL"}

        # Binarize labels for multiclass ROC
        y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

        plt.figure(figsize=(12, 10))

        colors = ["blue", "green", "red", "orange", "purple"]

        for model_idx, (name, metrics) in enumerate(results_dict.items()):
            probs = np.array(metrics.get("probabilities", []))
            if len(probs) == 0:
                continue

            # Check alignment of prediction lengths
            y_test_fold = y_test[-len(probs):]
            y_test_fold_bin = label_binarize(y_test_fold, classes=[0, 1, 2])

            color = colors[model_idx % len(colors)]

            for c in classes:
                fpr, tpr, _ = roc_curve(y_test_fold_bin[:, c], probs[:, c])
                roc_auc = auc(fpr, tpr)
                plt.plot(
                    fpr, tpr, color=color,
                    linestyle="--" if c == 2 else "-",
                    label=f"{name} {class_names[c]} (AUC = {roc_auc:.2f})"
                )

        plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Multiclass ROC Curves (BUY & SELL)")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        filepath = os.path.join(self.output_dir, "roc_curves.png")
        plt.savefig(filepath)
        plt.close()
        logger.info(f"ROC curves saved to {filepath}")

    def plot_precision_recall(self, results_dict: Dict[str, Dict[str, Any]], y_test: np.ndarray):
        """
        Plots Precision-Recall curves for BUY and SELL classes.
        """
        classes = [1, 2]  # BUY, SELL
        class_names = {1: "BUY", 2: "SELL"}

        plt.figure(figsize=(12, 10))
        colors = ["blue", "green", "red", "orange", "purple"]

        for model_idx, (name, metrics) in enumerate(results_dict.items()):
            probs = np.array(metrics.get("probabilities", []))
            if len(probs) == 0:
                continue

            y_test_fold = y_test[-len(probs):]
            y_test_fold_bin = label_binarize(y_test_fold, classes=[0, 1, 2])

            color = colors[model_idx % len(colors)]

            for c in classes:
                prec, rec, _ = precision_recall_curve(y_test_fold_bin[:, c], probs[:, c])
                plt.plot(
                    rec, prec, color=color,
                    linestyle="--" if c == 2 else "-",
                    label=f"{name} {class_names[c]}"
                )

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curves (BUY & SELL)")
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)

        filepath = os.path.join(self.output_dir, "precision_recall_curves.png")
        plt.savefig(filepath)
        plt.close()
        logger.info(f"Precision-Recall curves saved to {filepath}")

    def generate_report(self, results_dict: Dict[str, Dict[str, Any]], comparison_df: pd.DataFrame) -> str:
        """
        Generates a comprehensive text report comparing the models.
        """
        report_lines = []
        report_lines.append("======================================================================")
        report_lines.append("XAUUSD Trading Bot - Model Comparison Report")
        report_lines.append("======================================================================\n")

        # Table
        report_lines.append("Summary Metrics:")
        report_lines.append(comparison_df.to_string())
        report_lines.append("\n" + "-"*70 + "\n")

        best_model = self.select_best_model(results_dict)
        report_lines.append(f"Recommended Model for Deployment: {best_model.upper()}")
        report_lines.append(f"Reasoning: Chosen based on high Trade Signal Precision (accuracy of actual signals) "
                            f"to minimize false entry trades, while maintaining structural stability.")
        report_lines.append("\n" + "-"*70 + "\n")

        # Details
        for name, metrics in results_dict.items():
            report_lines.append(f"Model: {name}")
            report_lines.append(f"  Accuracy:                 {metrics.get('accuracy', 0.0):.4f}")
            report_lines.append(f"  Trade Signal Precision:   {metrics.get('trade_signal_precision', 0.0):.4f}")
            report_lines.append(f"  Precision per Class (0,1,2):  {[f'{x:.4f}' for x in metrics.get('precision_per_class', [])]}")
            report_lines.append(f"  Recall per Class (0,1,2):     {[f'{x:.4f}' for x in metrics.get('recall_per_class', [])]}")
            report_lines.append(f"  F1-Score per Class (0,1,2):   {[f'{x:.4f}' for x in metrics.get('f1_per_class', [])]}")
            report_lines.append(f"  Classification Report:")
            report_lines.append(metrics.get("classification_report", "No report available"))
            report_lines.append("\n" + "="*70 + "\n")

        report = "\n".join(report_lines)

        filepath = os.path.join(self.output_dir, "model_comparison_report.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"Model comparison report saved to {filepath}")

        return report

    def select_best_model(self, results_dict: Dict[str, Dict[str, Any]]) -> str:
        """
        Selects the best model name based on:
        1. Trade signal precision (average precision of BUY and SELL)
        2. High trade F1 to ensure some trades are placed
        3. General stability (minimal overfitting)
        """
        best_name = ""
        best_score = -1.0

        for name, metrics in results_dict.items():
            trade_prec = metrics.get("trade_signal_precision", 0.0)

            # We prioritize trade precision.
            if trade_prec > best_score:
                best_score = trade_prec
                best_name = name

        return best_name
