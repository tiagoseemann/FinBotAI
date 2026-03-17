"""
LLM response evaluation utilities.
Measures quality of agent responses using ROUGE, sentiment, and LLM-based evaluation.
"""

import json
import re
from typing import Dict, Any, List, Optional

from rouge_score import rouge_scorer
from textblob import TextBlob

from src.config import settings
from src.llm.prompts import format_evaluation_prompt
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ResponseEvaluator:
    """
    Evaluate quality of LLM responses.

    Metrics:
    - ROUGE scores (similarity to expected responses)
    - Sentiment analysis
    - Length and formatting
    - LLM-based relevance scoring
    """

    def __init__(self, llm_client=None):
        """
        Initialize evaluator.

        Args:
            llm_client: Optional LLM client for LLM-based evaluation
        """
        self.rouge_scorer = rouge_scorer.RougeScorer(
            settings.eval_rouge_types,
            use_stemmer=True
        )
        self.llm_client = llm_client

    def calculate_rouge_scores(
        self,
        generated: str,
        reference: str
    ) -> Dict[str, float]:
        """
        Calculate ROUGE scores between generated and reference text.

        Args:
            generated: Generated response
            reference: Reference (expected) response

        Returns:
            Dict with rouge1, rouge2, rougeL F-scores
        """
        scores = self.rouge_scorer.score(reference, generated)

        return {
            metric: scores[metric].fmeasure
            for metric in settings.eval_rouge_types
        }

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text.

        Args:
            text: Text to analyze

        Returns:
            Dict with polarity and subjectivity scores
        """
        try:
            blob = TextBlob(text)
            return {
                "polarity": blob.sentiment.polarity,  # -1 (negative) to 1 (positive)
                "subjectivity": blob.sentiment.subjectivity  # 0 (objective) to 1 (subjective)
            }
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return {"polarity": 0.0, "subjectivity": 0.5}

    def evaluate_response_quality(
        self,
        response: str,
        reference: Optional[str] = None,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive response quality evaluation.

        Args:
            response: Generated response
            reference: Optional reference response for ROUGE
            context: Optional context for evaluation

        Returns:
            Dict with various quality metrics
        """
        metrics = {}

        # Basic metrics
        metrics["length"] = len(response)
        metrics["word_count"] = len(response.split())
        metrics["has_emoji"] = bool(re.search(r'[\U0001F300-\U0001F9FF]', response))
        metrics["has_question"] = "?" in response
        metrics["has_exclamation"] = "!" in response

        # Sentiment
        sentiment = self.analyze_sentiment(response)
        metrics["sentiment_polarity"] = sentiment["polarity"]
        metrics["sentiment_subjectivity"] = sentiment["subjectivity"]

        # ROUGE scores (if reference provided)
        if reference:
            rouge_scores = self.calculate_rouge_scores(response, reference)
            metrics.update(rouge_scores)

        # Format check (WhatsApp-style should be short)
        metrics["is_concise"] = metrics["word_count"] <= 50  # 2-3 lines typically

        # Politeness indicators
        politeness_words = ["por favor", "obrigado", "desculpe", "agradeço"]
        metrics["politeness_score"] = sum(
            1 for word in politeness_words if word in response.lower()
        )

        return metrics

    def llm_evaluate_relevance(
        self,
        user_message: str,
        assistant_response: str,
        context: str = ""
    ) -> Optional[Dict[str, Any]]:
        """
        Use LLM to evaluate response relevance and quality.

        Args:
            user_message: User's message
            assistant_response: Assistant's response
            context: Conversation context

        Returns:
            Dict with scores or None if LLM not available
        """
        if not self.llm_client:
            logger.warning("LLM client not available for evaluation")
            return None

        try:
            # Format evaluation prompt
            eval_prompt = format_evaluation_prompt(
                context=context or "Primeira mensagem da conversa",
                user_message=user_message,
                assistant_response=assistant_response
            )

            # Get evaluation from LLM
            provider = settings.llm_provider.lower()

            if provider == "anthropic":
                response = self.llm_client.messages.create(
                    model=settings.llm_model,
                    max_tokens=500,
                    temperature=0.0,  # Deterministic evaluation
                    messages=[{"role": "user", "content": eval_prompt}]
                )
                eval_text = response.content[0].text
            else:  # OpenAI or Ollama
                response = self.llm_client.chat.completions.create(
                    model=settings.llm_model,
                    messages=[{"role": "user", "content": eval_prompt}],
                    temperature=0.0,
                    max_tokens=500
                )
                eval_text = response.choices[0].message.content

            # Parse JSON response
            # Extract JSON from markdown code blocks if present
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', eval_text, re.DOTALL)
            if json_match:
                eval_text = json_match.group(1)

            eval_result = json.loads(eval_text)
            return eval_result

        except Exception as e:
            logger.error(f"LLM evaluation failed: {e}")
            return None

    def evaluate_conversation(
        self,
        conversation_history: List[Dict[str, str]],
        expected_outcomes: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate an entire conversation.

        Args:
            conversation_history: List of messages
            expected_outcomes: Optional dict with expected results

        Returns:
            Dict with conversation-level metrics
        """
        assistant_messages = [m for m in conversation_history if m["role"] == "assistant"]

        if not assistant_messages:
            return {"error": "No assistant messages found"}

        # Aggregate metrics
        total_length = sum(len(m["text"]) for m in assistant_messages)
        avg_length = total_length / len(assistant_messages)

        sentiments = [self.analyze_sentiment(m["text"]) for m in assistant_messages]
        avg_polarity = sum(s["polarity"] for s in sentiments) / len(sentiments)

        # Response times (if timestamps available)
        response_count = len(assistant_messages)

        metrics = {
            "total_responses": response_count,
            "avg_response_length": avg_length,
            "total_length": total_length,
            "avg_sentiment_polarity": avg_polarity,
            "positive_responses": sum(1 for s in sentiments if s["polarity"] > 0.1),
            "neutral_responses": sum(1 for s in sentiments if -0.1 <= s["polarity"] <= 0.1),
            "negative_responses": sum(1 for s in sentiments if s["polarity"] < -0.1),
        }

        # Check expected outcomes
        if expected_outcomes:
            if "should_convert" in expected_outcomes:
                # Check if conversation led to expected outcome
                last_messages = " ".join(m["text"] for m in assistant_messages[-2:]).lower()
                conversion_keywords = ["link", "contratar", "finalizar", "cadastro"]
                has_conversion_signal = any(kw in last_messages for kw in conversion_keywords)
                metrics["conversion_signal_present"] = has_conversion_signal
                metrics["matches_expected_conversion"] = (
                    has_conversion_signal == expected_outcomes["should_convert"]
                )

        return metrics


def evaluate_test_set(
    test_conversations: List[Dict[str, Any]],
    agent: Any
) -> Dict[str, Any]:
    """
    Evaluate agent on a test set of conversations.

    Args:
        test_conversations: List of test conversation dicts
        agent: SalesAgent instance

    Returns:
        Aggregate evaluation metrics
    """
    evaluator = ResponseEvaluator(llm_client=agent.llm_client)

    all_metrics = []
    all_rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    all_latencies = []
    all_lead_scores = []

    logger.info(f"Evaluating agent on {len(test_conversations)} conversations...")

    for i, test_conv in enumerate(test_conversations):
        # Reset agent for each conversation
        agent.reset_conversation()

        conv_metrics = {
            "conversation_id": test_conv.get("id", f"test_{i}"),
            "messages": []
        }

        # Process each message
        for msg in test_conv["messages"]:
            if msg["role"] == "user":
                result = agent.process_message(msg["text"])

                # Evaluate response
                response_metrics = evaluator.evaluate_response_quality(
                    response=result["response"],
                    reference=msg.get("expected_response")
                )

                # Track metrics
                conv_metrics["messages"].append({
                    "user_message": msg["text"],
                    "agent_response": result["response"],
                    "metrics": response_metrics,
                    "latency_ms": result["latency_ms"],
                    "lead_score": result["lead_score"]
                })

                all_latencies.append(result["latency_ms"])
                all_lead_scores.append(result["lead_score"])

                # Aggregate ROUGE scores
                for metric in ["rouge1", "rouge2", "rougeL"]:
                    if metric in response_metrics:
                        all_rouge_scores[metric].append(response_metrics[metric])

        all_metrics.append(conv_metrics)

        if (i + 1) % 5 == 0:
            logger.info(f"Evaluated {i + 1}/{len(test_conversations)} conversations")

    # Calculate aggregate statistics
    aggregate = {
        "total_conversations": len(test_conversations),
        "total_messages": sum(len(m["messages"]) for m in all_metrics),

        # Latency stats
        "latency_mean_ms": sum(all_latencies) / len(all_latencies) if all_latencies else 0,
        "latency_p50_ms": sorted(all_latencies)[len(all_latencies) // 2] if all_latencies else 0,
        "latency_p95_ms": sorted(all_latencies)[int(len(all_latencies) * 0.95)] if all_latencies else 0,
        "latency_p99_ms": sorted(all_latencies)[int(len(all_latencies) * 0.99)] if all_latencies else 0,

        # Lead score stats
        "lead_score_mean": sum(all_lead_scores) / len(all_lead_scores) if all_lead_scores else 0,
        "lead_score_min": min(all_lead_scores) if all_lead_scores else 0,
        "lead_score_max": max(all_lead_scores) if all_lead_scores else 0,

        # ROUGE scores
        "rouge1_mean": sum(all_rouge_scores["rouge1"]) / len(all_rouge_scores["rouge1"]) if all_rouge_scores["rouge1"] else 0,
        "rouge2_mean": sum(all_rouge_scores["rouge2"]) / len(all_rouge_scores["rouge2"]) if all_rouge_scores["rouge2"] else 0,
        "rougeL_mean": sum(all_rouge_scores["rougeL"]) / len(all_rouge_scores["rougeL"]) if all_rouge_scores["rougeL"] else 0,

        "detailed_metrics": all_metrics
    }

    return aggregate


if __name__ == "__main__":
    # Test evaluator
    evaluator = ResponseEvaluator()

    # Test response
    response = "Ótimo! Temos o Crédito Flow com até R$ 50 mil. Taxa a partir de 1,99% ao mês. Quer saber mais? 😊"
    reference = "Temos empréstimos de até R$ 50 mil com taxas competitivas. Posso te explicar?"

    metrics = evaluator.evaluate_response_quality(response, reference)

    print("=== Response Evaluation ===")
    print(f"Response: {response}")
    print(f"\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
