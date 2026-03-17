"""
Metrics calculation for ML model evaluation.
Provides comprehensive performance metrics for binary classification.
"""

from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.logger import get_logger

logger = get_logger(__name__)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.

    Args:
        y_true: True labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
        y_proba: Predicted probabilities for positive class (optional)

    Returns:
        Dict of metric names to values
    """
    metrics = {}

    # Basic classification metrics
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)

    # AUC (requires probabilities)
    if y_proba is not None:
        try:
            metrics["auc"] = roc_auc_score(y_true, y_proba)
        except ValueError as e:
            logger.warning(f"Could not calculate AUC: {e}")
            metrics["auc"] = 0.0
    else:
        metrics["auc"] = 0.0

    # Confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics["true_positives"] = int(tp)
    metrics["true_negatives"] = int(tn)
    metrics["false_positives"] = int(fp)
    metrics["false_negatives"] = int(fn)

    # Derived metrics
    metrics["accuracy"] = (tp + tn) / (tp + tn + fp + fn)
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return metrics


def print_metrics_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    dataset_name: str = "Test"
) -> Dict[str, float]:
    """
    Calculate and print a formatted metrics report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
        dataset_name: Name of dataset for report header

    Returns:
        Dict of calculated metrics
    """
    metrics = calculate_metrics(y_true, y_pred, y_proba)

    logger.info(f"\n{'='*50}")
    logger.info(f"{dataset_name} Set Metrics")
    logger.info(f"{'='*50}")
    logger.info(f"AUC:        {metrics['auc']:.4f}")
    logger.info(f"Precision:  {metrics['precision']:.4f}")
    logger.info(f"Recall:     {metrics['recall']:.4f}")
    logger.info(f"F1 Score:   {metrics['f1']:.4f}")
    logger.info(f"Accuracy:   {metrics['accuracy']:.4f}")
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  TN: {metrics['true_negatives']:4d}  FP: {metrics['false_positives']:4d}")
    logger.info(f"  FN: {metrics['false_negatives']:4d}  TP: {metrics['true_positives']:4d}")
    logger.info(f"{'='*50}\n")

    return metrics


def get_feature_importance(
    model: Any,
    feature_names: list[str],
    top_n: int = 10
) -> pd.DataFrame:
    """
    Get feature importance from a trained model.

    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_n: Number of top features to return

    Returns:
        DataFrame with feature names and importance scores
    """
    if not hasattr(model, 'feature_importances_'):
        logger.warning("Model does not have feature_importances_ attribute")
        return pd.DataFrame()

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })

    # Sort by importance
    importance_df = importance_df.sort_values('importance', ascending=False)

    # Get top N
    if top_n:
        importance_df = importance_df.head(top_n)

    return importance_df


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Plot confusion matrix heatmap.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Optional path to save figure
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Not Converted', 'Converted'],
        yticklabels=['Not Converted', 'Converted']
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {save_path}")

    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Plot ROC curve.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities for positive class
        save_path: Optional path to save figure
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved ROC curve to {save_path}")

    plt.close()


def plot_feature_importance(
    importance_df: pd.DataFrame,
    save_path: Optional[str] = None
):
    """
    Plot feature importance bar chart.

    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()  # Highest at top
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved feature importance plot to {save_path}")

    plt.close()


def generate_evaluation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    feature_importance_df: pd.DataFrame,
    save_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a comprehensive evaluation report with plots.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        feature_importance_df: DataFrame with feature importance
        save_dir: Directory to save plots (optional)

    Returns:
        Dict with all metrics and plots
    """
    report = {}

    # Calculate metrics
    report["metrics"] = calculate_metrics(y_true, y_pred, y_proba)

    # Feature importance
    report["feature_importance"] = feature_importance_df.to_dict('records')

    # Generate plots if save_dir provided
    if save_dir:
        from pathlib import Path
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        plot_confusion_matrix(y_true, y_pred, save_path=str(save_dir / "confusion_matrix.png"))
        plot_roc_curve(y_true, y_proba, save_path=str(save_dir / "roc_curve.png"))
        plot_feature_importance(feature_importance_df, save_path=str(save_dir / "feature_importance.png"))

        logger.info(f"Saved evaluation plots to {save_dir}")

    return report


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)

    n_samples = 100
    y_true = np.random.randint(0, 2, n_samples)
    y_proba = np.random.rand(n_samples)
    y_pred = (y_proba > 0.5).astype(int)

    # Calculate and print metrics
    metrics = print_metrics_report(y_true, y_pred, y_proba, dataset_name="Test")

    # Test feature importance
    feature_names = [f"feature_{i}" for i in range(10)]
    importance = np.random.rand(10)

    class MockModel:
        def __init__(self, importance):
            self.feature_importances_ = importance

    model = MockModel(importance)
    importance_df = get_feature_importance(model, feature_names, top_n=5)

    print("\n=== Top 5 Features ===")
    print(importance_df)
