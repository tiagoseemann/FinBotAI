"""
Tests for ML modules.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import joblib

from src.ml.model import LeadScorer
from src.ml.metrics import calculate_metrics, get_feature_importance
from sklearn.ensemble import RandomForestClassifier


class TestLeadScorer:
    """Test LeadScorer model wrapper."""

    def test_scorer_initialization(self):
        """Test scorer can be initialized."""
        scorer = LeadScorer()
        assert scorer is not None

    def test_save_and_load_model(self):
        """Test model saving and loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pkl"
            scaler_path = Path(tmpdir) / "scaler.pkl"
            feature_names_path = Path(tmpdir) / "features.json"

            # Create dummy model
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler

            model = RandomForestClassifier(n_estimators=10, random_state=42)
            X_dummy = np.random.rand(50, 5)
            y_dummy = np.random.randint(0, 2, 50)
            model.fit(X_dummy, y_dummy)

            scaler = StandardScaler()
            scaler.fit(X_dummy)

            feature_names = [f"feature_{i}" for i in range(5)]

            # Save
            scorer = LeadScorer(
                model_path=model_path,
                scaler_path=scaler_path,
                feature_names_path=feature_names_path
            )
            scorer.save(model, scaler, feature_names)

            assert model_path.exists()
            assert scaler_path.exists()
            assert feature_names_path.exists()

            # Load
            scorer2 = LeadScorer(
                model_path=model_path,
                scaler_path=scaler_path,
                feature_names_path=feature_names_path
            )
            scorer2.load()

            assert scorer2.is_loaded
            assert scorer2.feature_names == feature_names

    def test_predict_proba(self):
        """Test probability prediction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Train simple model
            X = np.random.rand(100, 5)
            y = (X[:, 0] > 0.5).astype(int)

            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X, y)

            # Save and load
            scorer = LeadScorer(model_path=Path(tmpdir) / "model.pkl")
            scorer.save(model, feature_names=[f"f{i}" for i in range(5)])
            scorer.load()

            # Predict
            X_test = pd.DataFrame(np.random.rand(10, 5), columns=[f"f{i}" for i in range(5)])
            probas = scorer.predict_proba(X_test)

            assert len(probas) == 10
            assert all(0 <= p <= 1 for p in probas)

    def test_score_lead(self):
        """Test lead scoring with feature dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Train simple model
            X = pd.DataFrame(np.random.rand(100, 3), columns=["f0", "f1", "f2"])
            y = (X["f0"] > 0.5).astype(int)

            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X, y)

            # Save and load
            scorer = LeadScorer(model_path=Path(tmpdir) / "model.pkl")
            scorer.save(model, feature_names=["f0", "f1", "f2"])
            scorer.load()

            # Score single lead
            result = scorer.score_lead({"f0": 0.8, "f1": 0.5, "f2": 0.3})

            assert "probability" in result
            assert "prediction" in result
            assert "confidence" in result
            assert "score_percentage" in result
            assert 0 <= result["probability"] <= 1


class TestMetrics:
    """Test metrics calculations."""

    def test_calculate_metrics(self):
        """Test basic metrics calculation."""
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.9, 0.8, 0.4, 0.3, 0.85, 0.6])

        metrics = calculate_metrics(y_true, y_pred, y_proba)

        assert "auc" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "accuracy" in metrics

        # Check values are in valid ranges
        assert 0 <= metrics["auc"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1

    def test_get_feature_importance(self):
        """Test feature importance extraction."""
        # Train simple model
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        feature_names = [f"feature_{i}" for i in range(5)]
        importance_df = get_feature_importance(model, feature_names, top_n=3)

        assert len(importance_df) == 3
        assert "feature" in importance_df.columns
        assert "importance" in importance_df.columns

        # Should be sorted by importance
        importances = importance_df["importance"].values
        assert all(importances[i] >= importances[i+1] for i in range(len(importances)-1))

    def test_metrics_with_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_proba = np.array([0.0, 0.0, 1.0, 1.0])

        metrics = calculate_metrics(y_true, y_pred, y_proba)

        assert metrics["auc"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["accuracy"] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
