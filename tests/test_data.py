"""
Tests for data pipeline modules.
"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import json

from src.data.generator import generate_conversation, generate_dataset
from src.data.extractor import FeatureExtractor, extract_features_from_file
from src.data.loader import DataLoader, load_and_prepare_data


class TestGenerator:
    """Test conversation generator."""

    def test_generate_conversation(self):
        """Test single conversation generation."""
        conv = generate_conversation(
            customer_name="Test Customer",
            product_category="crédito",
            will_convert=True,
            engagement_level=0.8
        )

        assert "conversation_id" in conv
        assert "messages" in conv
        assert "metadata" in conv
        assert len(conv["messages"]) > 0
        assert conv["metadata"]["converted"] == True

    def test_generate_dataset(self):
        """Test dataset generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_conversations.json"

            conversations = generate_dataset(
                n_conversations=10,
                conversion_rate=0.5,
                output_path=output_path
            )

            assert len(conversations) == 10
            assert output_path.exists()

            # Check conversion rate is roughly correct
            conversions = sum(1 for c in conversations if c["metadata"]["converted"])
            assert 0 <= conversions <= 10

    def test_conversation_structure(self):
        """Test conversation has required fields."""
        conv = generate_conversation(
            customer_name="Test",
            product_category="investimento",
            will_convert=False,
            engagement_level=0.3
        )

        # Check required fields
        assert "conversation_id" in conv
        assert "customer_name" in conv
        assert "duration_seconds" in conv
        assert "messages" in conv
        assert "metadata" in conv

        # Check metadata
        metadata = conv["metadata"]
        assert "response_time_avg" in metadata
        assert "emoji_count" in metadata
        assert "converted" in metadata


class TestExtractor:
    """Test feature extractor."""

    def test_extract_from_conversation(self):
        """Test feature extraction from single conversation."""
        conv = generate_conversation(
            customer_name="Test",
            product_category="crédito",
            will_convert=True,
            engagement_level=0.8
        )

        extractor = FeatureExtractor()
        features = extractor.extract_from_conversation(conv)

        # Check key features exist
        assert "avg_response_time" in features
        assert "message_count" in features
        assert "emoji_count" in features
        assert "engagement_score" in features
        assert "sentiment_polarity" in features

        # Check values are reasonable
        assert features["message_count"] >= 0
        assert 0 <= features["engagement_score"] <= 1
        assert -1 <= features["sentiment_polarity"] <= 1

    def test_extract_from_dataset(self):
        """Test batch feature extraction."""
        conversations = generate_dataset(n_conversations=20, output_path=None)

        extractor = FeatureExtractor()
        df = extractor.extract_from_dataset(conversations)

        assert len(df) == 20
        assert "converted" in df.columns
        assert "conversation_id" in df.columns
        assert len(df.columns) > 10  # Should have many features

    def test_feature_types(self):
        """Test that features have correct types."""
        conv = generate_conversation(
            customer_name="Test",
            product_category="seguro",
            will_convert=False,
            engagement_level=0.5
        )

        extractor = FeatureExtractor()
        features = extractor.extract_from_conversation(conv)

        # All features should be numeric
        for key, value in features.items():
            assert isinstance(value, (int, float)), f"{key} is not numeric"


class TestLoader:
    """Test data loader."""

    def test_save_and_load_features(self):
        """Test saving and loading features to/from DuckDB."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.duckdb"
            loader = DataLoader(db_path=db_path)

            # Create test dataframe
            df = pd.DataFrame({
                "conversation_id": ["conv_1", "conv_2", "conv_3"],
                "feature1": [1.0, 2.0, 3.0],
                "feature2": [4.0, 5.0, 6.0],
                "converted": [0, 1, 1]
            })

            # Save and load
            loader.save_features(df, table_name="test_features")
            loaded_df = loader.load_features(table_name="test_features")

            assert len(loaded_df) == 3
            assert list(loaded_df.columns) == list(df.columns)

    def test_prepare_train_test_split(self):
        """Test train/test split preparation."""
        # Create test dataframe
        df = pd.DataFrame({
            "conversation_id": [f"conv_{i}" for i in range(100)],
            "feature1": range(100),
            "feature2": range(100, 200),
            "converted": [0, 1] * 50
        })

        loader = DataLoader()
        X_train, X_test, y_train, y_test = loader.prepare_train_test_split(
            df,
            test_size=0.2,
            random_state=42
        )

        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20

        # Features should not include ID or target
        assert "conversation_id" not in X_train.columns
        assert "converted" not in X_train.columns

    def test_normalize_features(self):
        """Test feature normalization."""
        # Create test data
        X_train = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [10, 20, 30, 40, 50]
        })

        X_test = pd.DataFrame({
            "feature1": [6, 7],
            "feature2": [60, 70]
        })

        loader = DataLoader()
        X_train_scaled, X_test_scaled, scaler = loader.normalize_features(
            X_train,
            X_test
        )

        # Check normalization (mean ~0, std ~1)
        assert abs(X_train_scaled["feature1"].mean()) < 0.01
        assert abs(X_train_scaled["feature1"].std() - 1.0) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
