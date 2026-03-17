"""
ML Model wrapper for lead scoring.
Handles model loading, prediction, and feature importance.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LeadScorer:
    """
    Wrapper for the lead scoring ML model.

    Handles:
    - Model loading
    - Feature preprocessing
    - Prediction with probability scores
    - Feature importance extraction
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        scaler_path: Optional[Path] = None,
        feature_names_path: Optional[Path] = None
    ):
        """
        Initialize lead scorer.

        Args:
            model_path: Path to trained model pickle
            scaler_path: Path to fitted scaler pickle
            feature_names_path: Path to feature names JSON
        """
        self.model_path = model_path or settings.model_path
        self.scaler_path = scaler_path or settings.scaler_path
        self.feature_names_path = feature_names_path or settings.feature_names_path

        self.model = None
        self.scaler = None
        self.feature_names = None

        # Load model artifacts if they exist
        if self.model_path.exists():
            self.load()

    def load(self):
        """Load model, scaler, and feature names from disk."""
        # Load model
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        logger.info(f"Loading model from {self.model_path}")
        self.model = joblib.load(self.model_path)

        # Load scaler
        if self.scaler_path.exists():
            logger.info(f"Loading scaler from {self.scaler_path}")
            self.scaler = joblib.load(self.scaler_path)
        else:
            logger.warning(f"Scaler not found at {self.scaler_path}, predictions may be inaccurate")

        # Load feature names
        if self.feature_names_path.exists():
            logger.info(f"Loading feature names from {self.feature_names_path}")
            with open(self.feature_names_path, 'r') as f:
                self.feature_names = json.load(f)
        else:
            logger.warning(f"Feature names not found at {self.feature_names_path}")

        logger.info("Model loaded successfully")

    def save(
        self,
        model: Any,
        scaler: Optional[StandardScaler] = None,
        feature_names: Optional[list[str]] = None
    ):
        """
        Save model artifacts to disk.

        Args:
            model: Trained model to save
            scaler: Fitted scaler to save
            feature_names: List of feature names
        """
        # Save model
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, self.model_path)
        logger.info(f"Saved model to {self.model_path}")

        self.model = model

        # Save scaler
        if scaler is not None:
            self.scaler_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(scaler, self.scaler_path)
            logger.info(f"Saved scaler to {self.scaler_path}")
            self.scaler = scaler

        # Save feature names
        if feature_names is not None:
            self.feature_names_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.feature_names_path, 'w') as f:
                json.dump(feature_names, f, indent=2)
            logger.info(f"Saved feature names to {self.feature_names_path}")
            self.feature_names = feature_names

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict probability of conversion.

        Args:
            X: Features (DataFrame or array)

        Returns:
            Array of probabilities for positive class (conversion)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")

        # Ensure features are in correct order
        if isinstance(X, pd.DataFrame):
            if self.feature_names:
                # Reorder columns to match training
                X = X[self.feature_names]
            X = X.values

        # Scale if scaler available
        if self.scaler is not None:
            X = self.scaler.transform(X)

        # Get probabilities for positive class
        probas = self.model.predict_proba(X)[:, 1]

        return probas

    def predict(self, X: Union[pd.DataFrame, np.ndarray], threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary labels (0 or 1).

        Args:
            X: Features
            threshold: Probability threshold for positive class

        Returns:
            Array of binary predictions
        """
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)

    def score_lead(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Score a single lead and return detailed results.

        Args:
            features: Dict of feature name -> value

        Returns:
            Dict with score, probability, confidence, and top features
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")

        # Convert to DataFrame
        df = pd.DataFrame([features])

        # Get probability
        proba = self.predict_proba(df)[0]

        # Get prediction
        prediction = int(proba >= 0.5)

        # Get confidence (distance from 0.5)
        confidence = abs(proba - 0.5) * 2  # Scale to 0-1

        # Get top contributing features (if possible)
        top_features = self._get_top_contributing_features(features)

        return {
            "probability": float(proba),
            "prediction": prediction,
            "confidence": float(confidence),
            "score_percentage": float(proba * 100),
            "top_features": top_features
        }

    def _get_top_contributing_features(
        self,
        features: Dict[str, float],
        top_n: int = 5
    ) -> list[Dict[str, Any]]:
        """
        Get top features contributing to the prediction.

        Uses model's feature importances as proxy for contribution.

        Args:
            features: Feature dict
            top_n: Number of top features to return

        Returns:
            List of dicts with feature name, value, and importance
        """
        if not hasattr(self.model, 'feature_importances_'):
            return []

        if not self.feature_names:
            return []

        # Get feature importances
        importances = self.model.feature_importances_

        # Create list of (feature_name, value, importance)
        feature_info = []
        for name, importance in zip(self.feature_names, importances):
            if name in features:
                feature_info.append({
                    "feature": name,
                    "value": float(features[name]),
                    "importance": float(importance)
                })

        # Sort by importance
        feature_info = sorted(feature_info, key=lambda x: x["importance"], reverse=True)

        return feature_info[:top_n]

    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get feature importance from the model.

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with feature names and importances
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        if not hasattr(self.model, 'feature_importances_'):
            logger.warning("Model does not have feature_importances_")
            return pd.DataFrame()

        if not self.feature_names:
            logger.warning("Feature names not available")
            return pd.DataFrame()

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        })

        importance_df = importance_df.sort_values('importance', ascending=False)

        if top_n:
            importance_df = importance_df.head(top_n)

        return importance_df

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None


if __name__ == "__main__":
    # Test model wrapper
    scorer = LeadScorer()

    if scorer.is_loaded:
        print("Model loaded successfully!")

        # Test with dummy features
        test_features = {
            "avg_response_time": 15.0,
            "message_count": 5,
            "emoji_count": 2,
            "question_count": 3,
            "engagement_score": 0.75,
            "sentiment_polarity": 0.5
        }

        # Get missing features with defaults
        if scorer.feature_names:
            for fname in scorer.feature_names:
                if fname not in test_features:
                    test_features[fname] = 0.0

        result = scorer.score_lead(test_features)

        print("\n=== Lead Scoring Result ===")
        print(f"Probability: {result['probability']:.3f}")
        print(f"Score: {result['score_percentage']:.1f}%")
        print(f"Prediction: {'Convert' if result['prediction'] else 'Not Convert'}")
        print(f"Confidence: {result['confidence']:.3f}")

        print("\nTop Contributing Features:")
        for feat in result['top_features']:
            print(f"  {feat['feature']}: {feat['value']} (importance: {feat['importance']:.3f})")

        # Feature importance
        importance_df = scorer.get_feature_importance(top_n=5)
        print("\n=== Top 5 Features ===")
        print(importance_df)
    else:
        print("No trained model found. Train a model first using scripts/train.py")
