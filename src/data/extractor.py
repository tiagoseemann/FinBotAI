"""
Extract features from conversations for ML model training.
Transforms raw conversation data into structured features.
"""

import json
import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import numpy as np
from textblob import TextBlob

from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureExtractor:
    """Extract features from conversation data."""

    def __init__(self):
        """Initialize feature extractor."""
        # Keywords for specific intents
        self.urgency_keywords = [
            "urgente", "rápido", "hoje", "agora", "já", "quando", "prazo"
        ]
        self.budget_keywords = [
            "quanto", "valor", "preço", "taxa", "custo", "caro", "barato", "juros"
        ]
        self.positive_keywords = [
            "sim", "quero", "interessei", "gostei", "perfeito", "ótimo", "bom"
        ]
        self.negative_keywords = [
            "não", "depois", "agora não", "sem interesse", "vou pensar"
        ]

    def extract_from_conversation(self, conversation: dict[str, Any]) -> dict[str, float]:
        """
        Extract features from a single conversation.

        Args:
            conversation: Conversation dict with messages and metadata

        Returns:
            Dict of feature name -> value
        """
        messages = conversation["messages"]
        metadata = conversation["metadata"]
        user_messages = [m for m in messages if m["role"] == "user"]

        # Combine all user text
        user_text = " ".join(m["text"] for m in user_messages).lower()
        user_text_length = len(user_text)

        # Behavioral Features
        features = {
            # Response time (faster = more engaged)
            "avg_response_time": metadata.get("response_time_avg", 60.0),
            "min_response_time": self._get_min_response_time(messages),

            # Message characteristics
            "message_count": len(user_messages),
            "avg_message_length": metadata.get("message_length_avg", 50.0),
            "total_message_length": user_text_length,

            # Engagement indicators
            "emoji_count": metadata.get("emoji_count", 0),
            "question_count": metadata.get("question_count", 0),
            "exclamation_count": user_text.count("!"),

            # Conversation duration
            "conversation_duration": conversation.get("duration_seconds", 0) / 60.0,  # in minutes
        }

        # Contextual Features
        features.update({
            # Product interest
            "product_credit": 1 if metadata.get("product_mentioned") == "crédito" else 0,
            "product_investment": 1 if metadata.get("product_mentioned") == "investimento" else 0,
            "product_insurance": 1 if metadata.get("product_mentioned") == "seguro" else 0,
            "product_card": 1 if metadata.get("product_mentioned") == "cartão" else 0,
            "product_retirement": 1 if metadata.get("product_mentioned") == "previdência" else 0,
            "product_consortium": 1 if metadata.get("product_mentioned") == "consórcio" else 0,

            # Intent indicators
            "mention_urgency": float(metadata.get("mention_urgency", False)),
            "mention_budget": float(metadata.get("mention_budget", False)),

            # Keyword counts
            "urgency_keyword_count": sum(1 for word in self.urgency_keywords if word in user_text),
            "budget_keyword_count": sum(1 for word in self.budget_keywords if word in user_text),
            "positive_keyword_count": sum(1 for word in self.positive_keywords if word in user_text),
            "negative_keyword_count": sum(1 for word in self.negative_keywords if word in user_text),
        })

        # Linguistic Features
        sentiment = self._calculate_sentiment(user_text)
        features.update({
            "sentiment_polarity": sentiment["polarity"],
            "sentiment_subjectivity": sentiment["subjectivity"],
        })

        # Interaction patterns
        features.update({
            "response_speed_score": self._calculate_response_speed_score(features["avg_response_time"]),
            "engagement_score": self._calculate_engagement_score(features),
        })

        return features

    def _get_min_response_time(self, messages: list[dict]) -> float:
        """Calculate minimum response time from user."""
        user_messages = [m for m in messages if m["role"] == "user"]
        if len(user_messages) < 2:
            return 60.0

        timestamps = [m["timestamp"] for m in user_messages]
        diffs = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
        return min(diffs) if diffs else 60.0

    def _calculate_sentiment(self, text: str) -> dict[str, float]:
        """Calculate sentiment scores using TextBlob."""
        try:
            blob = TextBlob(text)
            return {
                "polarity": blob.sentiment.polarity,  # -1 (negative) to 1 (positive)
                "subjectivity": blob.sentiment.subjectivity  # 0 (objective) to 1 (subjective)
            }
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return {"polarity": 0.0, "subjectivity": 0.5}

    def _calculate_response_speed_score(self, avg_response_time: float) -> float:
        """
        Convert response time to a score (0-1, higher is better).

        Fast responses (< 10s) = high score
        Slow responses (> 60s) = low score
        """
        if avg_response_time < 10:
            return 1.0
        elif avg_response_time < 30:
            return 0.7
        elif avg_response_time < 60:
            return 0.4
        else:
            return 0.1

    def _calculate_engagement_score(self, features: dict) -> float:
        """
        Calculate overall engagement score from multiple signals.

        Considers: response speed, message count, emojis, questions
        """
        score = 0.0

        # Response speed component (40%)
        score += features["response_speed_score"] * 0.4

        # Activity component (30%)
        message_score = min(features["message_count"] / 8.0, 1.0)  # Normalize to max 8 messages
        score += message_score * 0.3

        # Enthusiasm component (30%)
        emoji_score = min(features["emoji_count"] / 3.0, 1.0)  # Normalize to max 3 emojis
        question_score = min(features["question_count"] / 3.0, 1.0)  # Normalize to max 3 questions
        enthusiasm = (emoji_score + question_score) / 2
        score += enthusiasm * 0.3

        return round(score, 3)

    def extract_from_dataset(
        self,
        conversations: list[dict[str, Any]],
        include_label: bool = True
    ) -> pd.DataFrame:
        """
        Extract features from a list of conversations.

        Args:
            conversations: List of conversation dicts
            include_label: Whether to include 'converted' label

        Returns:
            DataFrame with features (and optionally label)
        """
        logger.info(f"Extracting features from {len(conversations)} conversations...")

        features_list = []
        for i, conv in enumerate(conversations):
            try:
                features = self.extract_from_conversation(conv)

                # Add label if requested
                if include_label:
                    features["converted"] = int(conv["metadata"].get("converted", False))

                # Add conversation ID for tracking
                features["conversation_id"] = conv["conversation_id"]

                features_list.append(features)

                if (i + 1) % 100 == 0:
                    logger.info(f"Extracted features from {i + 1}/{len(conversations)} conversations")

            except Exception as e:
                logger.error(f"Failed to extract features from conversation {conv.get('conversation_id')}: {e}")
                continue

        df = pd.DataFrame(features_list)

        # Move conversation_id and label to front
        cols = ["conversation_id"]
        if include_label and "converted" in df.columns:
            cols.append("converted")
        cols += [c for c in df.columns if c not in cols]
        df = df[cols]

        logger.info(f"Extracted {len(df)} feature sets with {len(df.columns)} columns")

        return df


def extract_features_from_file(
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Load conversations from JSON and extract features.

    Args:
        input_path: Path to conversations JSON file
        output_path: Optional path to save features CSV

    Returns:
        DataFrame with extracted features
    """
    if input_path is None:
        input_path = settings.synthetic_conversations_path

    if not input_path.exists():
        raise FileNotFoundError(f"Conversations file not found: {input_path}")

    logger.info(f"Loading conversations from {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        conversations = json.load(f)

    # Extract features
    extractor = FeatureExtractor()
    df = extractor.extract_from_dataset(conversations)

    # Save if output path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved features to {output_path}")

    return df


if __name__ == "__main__":
    # Test feature extraction
    from src.data.generator import generate_conversation

    # Generate a sample conversation
    sample_conv = generate_conversation(
        customer_name="João Silva",
        product_category="crédito",
        will_convert=True,
        engagement_level=0.8
    )

    # Extract features
    extractor = FeatureExtractor()
    features = extractor.extract_from_conversation(sample_conv)

    print("\n=== Extracted Features ===")
    for key, value in features.items():
        print(f"{key:30s}: {value}")

    # Test on full dataset (if exists)
    if settings.synthetic_conversations_path.exists():
        df = extract_features_from_file()
        print(f"\n=== Dataset Summary ===")
        print(df.describe())
        print(f"\nConversion rate: {df['converted'].mean():.2%}")
