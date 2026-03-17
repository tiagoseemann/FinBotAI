"""
Data loading and persistence utilities using DuckDB.
Handles train/test splits and feature normalization.
"""

from pathlib import Path
from typing import Optional, Tuple

import duckdb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """Handle data loading, saving, and transformations."""

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize data loader.

        Args:
            db_path: Path to DuckDB database file
        """
        self.db_path = db_path or settings.features_db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def save_features(
        self,
        df: pd.DataFrame,
        table_name: str = "features"
    ):
        """
        Save features DataFrame to DuckDB.

        Args:
            df: Features DataFrame
            table_name: Name of table to create/replace
        """
        logger.info(f"Saving {len(df)} rows to {table_name} in {self.db_path}")

        with duckdb.connect(str(self.db_path)) as conn:
            # Create or replace table
            conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df")
            logger.info(f"Saved to table '{table_name}'")

    def load_features(
        self,
        table_name: str = "features",
        query: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load features from DuckDB.

        Args:
            table_name: Name of table to load
            query: Optional SQL query (if None, loads entire table)

        Returns:
            Features DataFrame
        """
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        logger.info(f"Loading data from {self.db_path}")

        with duckdb.connect(str(self.db_path)) as conn:
            if query:
                df = conn.execute(query).df()
            else:
                df = conn.execute(f"SELECT * FROM {table_name}").df()

            logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
            return df

    def prepare_train_test_split(
        self,
        df: pd.DataFrame,
        target_col: str = "converted",
        test_size: float = 0.2,
        random_state: int = 42,
        exclude_cols: Optional[list[str]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare train/test split for ML.

        Args:
            df: Features DataFrame with target column
            target_col: Name of target column
            test_size: Proportion for test set
            random_state: Random seed
            exclude_cols: Columns to exclude from features (e.g., IDs)

        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info(f"Preparing train/test split (test_size={test_size})")

        # Separate features and target
        exclude_cols = exclude_cols or ["conversation_id"]
        feature_cols = [c for c in df.columns if c != target_col and c not in exclude_cols]

        X = df[feature_cols]
        y = df[target_col]

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y  # Maintain class balance
        )

        logger.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
        logger.info(f"Train conversion rate: {y_train.mean():.2%}")
        logger.info(f"Test conversion rate: {y_test.mean():.2%}")

        return X_train, X_test, y_train, y_test

    def normalize_features(
        self,
        X_train: pd.DataFrame,
        X_test: Optional[pd.DataFrame] = None,
        scaler_path: Optional[Path] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], StandardScaler]:
        """
        Normalize features using StandardScaler.

        Args:
            X_train: Training features
            X_test: Optional test features
            scaler_path: Optional path to save scaler

        Returns:
            X_train_scaled, X_test_scaled, scaler
        """
        logger.info("Normalizing features with StandardScaler")

        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )

        X_test_scaled = None
        if X_test is not None:
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )

        # Save scaler if path provided
        if scaler_path:
            scaler_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(scaler, scaler_path)
            logger.info(f"Saved scaler to {scaler_path}")

        return X_train_scaled, X_test_scaled, scaler

    def load_scaler(self, scaler_path: Optional[Path] = None) -> StandardScaler:
        """
        Load a saved scaler.

        Args:
            scaler_path: Path to scaler file

        Returns:
            Loaded scaler
        """
        scaler_path = scaler_path or settings.scaler_path

        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")

        logger.info(f"Loading scaler from {scaler_path}")
        return joblib.load(scaler_path)

    def get_feature_statistics(self, df: pd.DataFrame) -> dict:
        """
        Get statistics about the features dataset.

        Args:
            df: Features DataFrame

        Returns:
            Dict with statistics
        """
        stats = {
            "n_samples": len(df),
            "n_features": len([c for c in df.columns if c not in ["conversation_id", "converted"]]),
            "conversion_rate": df["converted"].mean() if "converted" in df.columns else None,
            "missing_values": df.isnull().sum().to_dict(),
            "feature_types": df.dtypes.to_dict()
        }

        if "converted" in df.columns:
            stats["class_distribution"] = df["converted"].value_counts().to_dict()

        return stats


def load_and_prepare_data(
    csv_path: Optional[Path] = None,
    test_size: float = 0.2,
    normalize: bool = True,
    random_state: int = 42
) -> dict:
    """
    Convenience function to load and prepare data in one call.

    Args:
        csv_path: Path to features CSV (if None, loads from DB)
        test_size: Proportion for test set
        normalize: Whether to normalize features
        random_state: Random seed

    Returns:
        Dict with X_train, X_test, y_train, y_test, scaler, feature_names
    """
    logger.info("Loading and preparing data...")

    # Load data
    if csv_path and csv_path.exists():
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} samples from CSV")
    else:
        loader = DataLoader()
        df = loader.load_features()

    # Split
    loader = DataLoader()
    X_train, X_test, y_train, y_test = loader.prepare_train_test_split(
        df,
        test_size=test_size,
        random_state=random_state
    )

    feature_names = X_train.columns.tolist()

    # Normalize if requested
    scaler = None
    if normalize:
        X_train, X_test, scaler = loader.normalize_features(
            X_train,
            X_test,
            scaler_path=settings.scaler_path
        )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": feature_names
    }


if __name__ == "__main__":
    # Test data loading
    from src.data.generator import generate_dataset
    from src.data.extractor import extract_features_from_file

    # Generate and extract if needed
    if not settings.synthetic_conversations_path.exists():
        print("Generating synthetic conversations...")
        generate_dataset(n_conversations=100)

    if not settings.features_db_path.exists():
        print("Extracting features...")
        df = extract_features_from_file()

        # Save to DuckDB
        loader = DataLoader()
        loader.save_features(df)

    # Load and prepare data
    data = load_and_prepare_data(test_size=0.2, normalize=True)

    print("\n=== Data Preparation Summary ===")
    print(f"Training samples: {len(data['X_train'])}")
    print(f"Test samples: {len(data['X_test'])}")
    print(f"Number of features: {len(data['feature_names'])}")
    print(f"Features: {data['feature_names'][:5]}...")

    # Show statistics
    loader = DataLoader()
    df = loader.load_features()
    stats = loader.get_feature_statistics(df)

    print("\n=== Dataset Statistics ===")
    print(f"Total samples: {stats['n_samples']}")
    print(f"Conversion rate: {stats['conversion_rate']:.2%}")
    print(f"Class distribution: {stats['class_distribution']}")
