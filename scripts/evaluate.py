#!/usr/bin/env python3
"""
Evaluation script for FinBot.

Runs comprehensive evaluation of both ML model and LLM agent.

Usage:
    python scripts/evaluate.py [OPTIONS]
"""

import argparse
import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import settings
from src.data.loader import load_and_prepare_data
from src.ml.model import LeadScorer
from src.ml.metrics import print_metrics_report, get_feature_importance, generate_evaluation_report
from src.llm.agent import SalesAgent
from src.llm.evaluator import evaluate_test_set
from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_ml_model():
    """Evaluate ML model performance."""
    logger.info("\n" + "=" * 70)
    logger.info("ML MODEL EVALUATION")
    logger.info("=" * 70)

    try:
        # Load model
        scorer = LeadScorer()
        if not scorer.is_loaded:
            logger.error("Model not found. Train model first: python scripts/train.py")
            return None

        # Load test data
        data = load_and_prepare_data(
            test_size=settings.test_size,
            normalize=True,
            random_state=settings.random_state
        )

        X_test = data['X_test']
        y_test = data['y_test']

        # Predictions
        y_pred_proba = scorer.predict_proba(X_test)
        y_pred = scorer.predict(X_test)

        # Calculate metrics
        metrics = print_metrics_report(y_test, y_pred, y_pred_proba, "Test Set")

        # Feature importance
        importance_df = get_feature_importance(
            scorer.model,
            scorer.feature_names,
            top_n=10
        )

        logger.info("\nTop 10 Features:")
        for _, row in importance_df.iterrows():
            logger.info(f"  {row['feature']:30s}: {row['importance']:.4f}")

        # Generate detailed report
        report = generate_evaluation_report(
            y_test,
            y_pred,
            y_pred_proba,
            importance_df,
            save_dir=str(settings.REPORTS_DIR / "evaluation")
        )

        logger.info(f"\nDetailed report saved to: {settings.REPORTS_DIR / 'evaluation'}")

        return {
            "metrics": metrics,
            "feature_importance": importance_df.to_dict('records'),
            "model_loaded": True
        }

    except Exception as e:
        logger.error(f"ML evaluation failed: {e}", exc_info=True)
        return None


def evaluate_llm_agent(n_test_conversations: int = 20):
    """Evaluate LLM agent performance."""
    logger.info("\n" + "=" * 70)
    logger.info("LLM AGENT EVALUATION")
    logger.info("=" * 70)

    try:
        # Initialize agent
        agent = SalesAgent(use_scoring=True)

        # Create test conversations
        test_conversations = [
            {
                "id": f"test_{i}",
                "messages": [
                    {
                        "role": "user",
                        "text": msg,
                        "expected_response": None  # We won't use ROUGE for now
                    }
                    for msg in test_msgs
                ]
            }
            for i, test_msgs in enumerate([
                ["Olá, preciso de dinheiro urgente", "Quanto custa?", "Pode ser rápido?"],
                ["Quero investir meu dinheiro", "Quais são as opções?"],
                ["Preciso de um seguro", "Quanto é?"],
                ["Olá", "Não tenho interesse"],
                ["Quero um cartão de crédito", "Sem anuidade?", "Perfeito!"],
            ] * (n_test_conversations // 5))
        ]

        # Evaluate
        logger.info(f"Testing agent with {len(test_conversations)} conversations...")
        results = evaluate_test_set(test_conversations[:n_test_conversations], agent)

        # Print results
        logger.info("\nAgent Performance:")
        logger.info(f"  Total Conversations: {results['total_conversations']}")
        logger.info(f"  Total Messages:      {results['total_messages']}")
        logger.info(f"  Latency Mean:        {results['latency_mean_ms']:.1f} ms")
        logger.info(f"  Latency P95:         {results['latency_p95_ms']:.1f} ms")
        logger.info(f"  Latency P99:         {results['latency_p99_ms']:.1f} ms")
        logger.info(f"  Lead Score Mean:     {results['lead_score_mean']:.3f}")

        # Check latency target
        if results['latency_p95_ms'] < 500:
            logger.info("\n✓ SUCCESS: P95 latency < 500ms target")
        else:
            logger.warning("\n⚠ WARNING: P95 latency above 500ms target")

        return results

    except Exception as e:
        logger.error(f"LLM evaluation failed: {e}", exc_info=True)
        return None


def integration_test():
    """Test full integration: features -> ML -> LLM."""
    logger.info("\n" + "=" * 70)
    logger.info("INTEGRATION TEST")
    logger.info("=" * 70)

    try:
        # Simulate a conversation
        agent = SalesAgent(use_scoring=True)

        test_messages = [
            "Olá! Preciso de R$ 20 mil com urgência",
            "Quais são os juros?",
            "Perfeito, quero contratar!"
        ]

        logger.info("Simulating conversation...")
        for i, msg in enumerate(test_messages, 1):
            logger.info(f"\n[Message {i}] User: {msg}")

            result = agent.process_message(msg)

            logger.info(f"Agent: {result['response']}")
            logger.info(f"Lead Score: {result['lead_score_percentage']:.1f}%")
            logger.info(f"Latency: {result['latency_ms']:.0f}ms")

            if result['should_recommend']:
                logger.info(f"Recommendation: {result['product_recommendation']}")

        logger.info("\n✓ Integration test passed")
        return True

    except Exception as e:
        logger.error(f"Integration test failed: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate FinBot system",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--ml-only",
        action="store_true",
        help="Only evaluate ML model"
    )

    parser.add_argument(
        "--llm-only",
        action="store_true",
        help="Only evaluate LLM agent"
    )

    parser.add_argument(
        "--n-conversations",
        type=int,
        default=20,
        help="Number of test conversations for LLM evaluation"
    )

    parser.add_argument(
        "--skip-integration",
        action="store_true",
        help="Skip integration test"
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("FinBot System Evaluation")
    logger.info("=" * 70)

    results = {}

    # ML evaluation
    if not args.llm_only:
        ml_results = evaluate_ml_model()
        results['ml'] = ml_results

    # LLM evaluation
    if not args.ml_only:
        llm_results = evaluate_llm_agent(args.n_conversations)
        results['llm'] = llm_results

    # Integration test
    if not args.skip_integration and not args.ml_only and not args.llm_only:
        integration_results = integration_test()
        results['integration'] = integration_results

    # Save results
    output_path = settings.REPORTS_DIR / "evaluation_results.json"
    with open(output_path, 'w') as f:
        # Convert to JSON-serializable format
        json_results = json.loads(json.dumps(results, default=str))
        json.dump(json_results, f, indent=2)

    logger.info(f"\n✓ Evaluation results saved to: {output_path}")

    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
