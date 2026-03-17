#!/usr/bin/env python3
"""
Training script for FinBot ML model.

Usage:
    python scripts/train.py [OPTIONS]

Examples:
    python scripts/train.py --n-trials 50 --optimize
    python scripts/train.py --no-optimize  # Use default params
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import settings
from src.data.generator import generate_dataset
from src.data.extractor import extract_features_from_file
from src.data.loader import DataLoader
from src.ml.training import train_model
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Train FinBot lead scoring model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials for hyperparameter optimization"
    )

    parser.add_argument(
        "--optimize",
        action="store_true",
        default=True,
        help="Run hyperparameter optimization"
    )

    parser.add_argument(
        "--no-optimize",
        action="store_false",
        dest="optimize",
        help="Skip optimization and use default parameters"
    )

    parser.add_argument(
        "--generate-data",
        action="store_true",
        help="Generate new synthetic data before training"
    )

    parser.add_argument(
        "--n-conversations",
        type=int,
        default=500,
        help="Number of conversations to generate"
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set size (0-1)"
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("FinBot Model Training")
    logger.info("=" * 70)
    logger.info(f"Optimize: {args.optimize}")
    logger.info(f"N Trials: {args.n_trials}")
    logger.info(f"Test Size: {args.test_size}")
    logger.info("=" * 70)

    try:
        # Step 1: Generate data if requested
        if args.generate_data:
            logger.info("\n[1/4] Generating synthetic data...")
            generate_dataset(
                n_conversations=args.n_conversations,
                conversion_rate=0.3
            )
        else:
            if not settings.synthetic_conversations_path.exists():
                logger.info("\n[1/4] No data found, generating synthetic data...")
                generate_dataset(
                    n_conversations=args.n_conversations,
                    conversion_rate=0.3
                )
            else:
                logger.info("\n[1/4] Using existing synthetic data")

        # Step 2: Extract features
        logger.info("\n[2/4] Extracting features...")
        df = extract_features_from_file()

        # Save to DuckDB
        loader = DataLoader()
        loader.save_features(df)

        logger.info(f"Extracted {len(df)} feature sets")
        logger.info(f"Conversion rate: {df['converted'].mean():.2%}")

        # Step 3: Train model
        logger.info("\n[3/4] Training model...")
        results = train_model(
            n_trials=args.n_trials,
            optimize=args.optimize,
            save_model=True
        )

        # Step 4: Print summary
        logger.info("\n[4/4] Training complete!")
        logger.info("=" * 70)
        logger.info("RESULTS SUMMARY")
        logger.info("=" * 70)

        test_metrics = results['test_metrics']
        logger.info(f"Test AUC:       {test_metrics['auc']:.4f}")
        logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
        logger.info(f"Test Recall:    {test_metrics['recall']:.4f}")
        logger.info(f"Test F1:        {test_metrics['f1']:.4f}")

        logger.info("\nTop 5 Features:")
        for _, row in results['feature_importance'].head(5).iterrows():
            logger.info(f"  {row['feature']:30s}: {row['importance']:.4f}")

        logger.info("\nModel Artifacts:")
        logger.info(f"  Model:    {settings.model_path}")
        logger.info(f"  Scaler:   {settings.scaler_path}")
        logger.info(f"  Features: {settings.feature_names_path}")
        logger.info(f"  Report:   {settings.REPORTS_DIR / 'training'}")

        logger.info("=" * 70)

        # Check if model meets target
        if test_metrics['auc'] >= 0.75:
            logger.info("✓ SUCCESS: Model meets target AUC ≥ 0.75")
            return 0
        else:
            logger.warning("⚠ WARNING: Model AUC below target (0.75)")
            logger.warning("Consider increasing n_trials or generating more data")
            return 0  # Still success, just below target

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
