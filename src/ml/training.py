"""
ML model training with hyperparameter optimization.
Uses Optuna for hyperparameter tuning and LightGBM for classification.
"""

from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.model_selection import cross_val_score
from optuna.samplers import TPESampler

from src.config import settings
from src.data.loader import load_and_prepare_data
from src.ml.model import LeadScorer
from src.ml.metrics import (
    calculate_metrics,
    print_metrics_report,
    get_feature_importance,
    generate_evaluation_report
)
from src.utils.logger import get_logger
from src.utils.timing import timer

logger = get_logger(__name__)


class ModelTrainer:
    """
    Handle model training with hyperparameter optimization.
    """

    def __init__(
        self,
        n_trials: int = 50,
        cv_folds: int = 5,
        random_state: int = 42,
        timeout: Optional[int] = None
    ):
        """
        Initialize trainer.

        Args:
            n_trials: Number of Optuna trials
            cv_folds: Number of cross-validation folds
            random_state: Random seed
            timeout: Optional timeout in seconds for optimization
        """
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.timeout = timeout

        self.best_params = None
        self.best_score = None
        self.study = None

    def objective(
        self,
        trial: optuna.Trial,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> float:
        """
        Optuna objective function for hyperparameter optimization.

        Args:
            trial: Optuna trial
            X_train: Training features
            y_train: Training labels

        Returns:
            Cross-validated AUC score
        """
        # Define hyperparameter search space
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'random_state': self.random_state,
            'n_jobs': -1,

            # Tunable parameters
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        }

        # Create model
        model = lgb.LGBMClassifier(**params)

        # Cross-validation
        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=self.cv_folds,
            scoring='roc_auc',
            n_jobs=-1
        )

        # Return mean AUC
        return cv_scores.mean()

    def optimize_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization with Optuna.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Dict with best parameters and score
        """
        logger.info(f"Starting hyperparameter optimization with {self.n_trials} trials...")

        # Create study
        sampler = TPESampler(seed=self.random_state)
        self.study = optuna.create_study(
            direction='maximize',
            sampler=sampler
        )

        # Optimize
        with timer("Hyperparameter optimization"):
            self.study.optimize(
                lambda trial: self.objective(trial, X_train, y_train),
                n_trials=self.n_trials,
                timeout=self.timeout,
                show_progress_bar=True
            )

        self.best_params = self.study.best_params
        self.best_score = self.study.best_value

        logger.info(f"Best AUC: {self.best_score:.4f}")
        logger.info(f"Best parameters: {self.best_params}")

        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_trials': len(self.study.trials)
        }

    def train_final_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        params: Optional[Dict[str, Any]] = None
    ) -> lgb.LGBMClassifier:
        """
        Train final model with best parameters.

        Args:
            X_train: Training features
            y_train: Training labels
            params: Model parameters (uses best_params if None)

        Returns:
            Trained model
        """
        if params is None:
            if self.best_params is None:
                raise ValueError("No parameters provided and optimization not run")
            params = self.best_params

        logger.info("Training final model with best parameters...")

        # Add fixed parameters
        full_params = {
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'random_state': self.random_state,
            'n_jobs': -1,
            **params
        }

        # Train model
        model = lgb.LGBMClassifier(**full_params)

        with timer("Model training"):
            model.fit(X_train, y_train)

        logger.info("Model training completed")

        return model

    def train_and_evaluate(
        self,
        optimize: bool = True,
        save_model: bool = True,
        save_report: bool = True
    ) -> Dict[str, Any]:
        """
        Complete training pipeline: load data, optimize, train, evaluate.

        Args:
            optimize: Whether to run hyperparameter optimization
            save_model: Whether to save trained model
            save_report: Whether to save evaluation report

        Returns:
            Dict with model, metrics, and artifacts
        """
        logger.info("=" * 60)
        logger.info("Starting training pipeline")
        logger.info("=" * 60)

        # Load and prepare data
        logger.info("Loading data...")
        data = load_and_prepare_data(
            test_size=settings.test_size,
            normalize=True,
            random_state=self.random_state
        )

        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']
        scaler = data['scaler']
        feature_names = data['feature_names']

        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        logger.info(f"Features: {len(feature_names)}")

        # Optimize hyperparameters
        if optimize:
            opt_result = self.optimize_hyperparameters(X_train, y_train)
            params = opt_result['best_params']
        else:
            # Use default parameters
            logger.info("Using default parameters (no optimization)")
            params = {
                'learning_rate': 0.05,
                'n_estimators': 200,
                'max_depth': 6,
                'num_leaves': 50,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
            }

        # Train final model
        model = self.train_final_model(X_train, y_train, params)

        # Evaluate on training set
        logger.info("\n=== Training Set Evaluation ===")
        y_train_pred = model.predict(X_train)
        y_train_proba = model.predict_proba(X_train)[:, 1]
        train_metrics = print_metrics_report(y_train, y_train_pred, y_train_proba, "Training")

        # Evaluate on test set
        logger.info("\n=== Test Set Evaluation ===")
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1]
        test_metrics = print_metrics_report(y_test, y_test_pred, y_test_proba, "Test")

        # Feature importance
        importance_df = get_feature_importance(model, feature_names, top_n=10)
        logger.info("\n=== Top 10 Features ===")
        for _, row in importance_df.iterrows():
            logger.info(f"  {row['feature']:30s}: {row['importance']:.4f}")

        # Save model
        if save_model:
            scorer = LeadScorer()
            scorer.save(model, scaler, feature_names)
            logger.info(f"Model saved to {scorer.model_path}")

        # Generate evaluation report
        report = None
        if save_report:
            report = generate_evaluation_report(
                y_test,
                y_test_pred,
                y_test_proba,
                importance_df,
                save_dir=str(settings.REPORTS_DIR / "training")
            )
            logger.info(f"Evaluation report saved to {settings.REPORTS_DIR / 'training'}")

        logger.info("=" * 60)
        logger.info("Training pipeline completed!")
        logger.info("=" * 60)

        return {
            'model': model,
            'scaler': scaler,
            'feature_names': feature_names,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'feature_importance': importance_df,
            'best_params': params,
            'report': report
        }


def train_model(
    n_trials: int = 50,
    optimize: bool = True,
    save_model: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to train a model.

    Args:
        n_trials: Number of Optuna trials
        optimize: Whether to run hyperparameter optimization
        save_model: Whether to save the model

    Returns:
        Training results
    """
    trainer = ModelTrainer(
        n_trials=n_trials,
        cv_folds=settings.cv_folds,
        random_state=settings.random_state,
        timeout=settings.optuna_timeout
    )

    results = trainer.train_and_evaluate(
        optimize=optimize,
        save_model=save_model,
        save_report=True
    )

    return results


if __name__ == "__main__":
    # Train model with default settings
    results = train_model(n_trials=30, optimize=True)

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Test AUC: {results['test_metrics']['auc']:.4f}")
    print(f"Test Precision: {results['test_metrics']['precision']:.4f}")
    print(f"Test Recall: {results['test_metrics']['recall']:.4f}")
    print(f"Test F1: {results['test_metrics']['f1']:.4f}")
    print("\nTop 5 Features:")
    for _, row in results['feature_importance'].head(5).iterrows():
        print(f"  {row['feature']:30s}: {row['importance']:.4f}")
    print("=" * 60)
