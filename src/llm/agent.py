"""
LLM-based sales agent for financial products.
Integrates ML lead scoring with conversational AI.
"""

import time
from typing import Dict, Any, List, Optional

from src.config import settings
from src.llm.prompts import (
    format_system_prompt,
    format_user_prompt,
    load_products,
    get_product_recommendation_prompt
)
from src.ml.model import LeadScorer
from src.data.extractor import FeatureExtractor
from src.utils.logger import get_logger
from src.utils.timing import timed

logger = get_logger(__name__)


class SalesAgent:
    """
    AI Sales Agent that combines ML lead scoring with LLM conversation.

    Features:
    - Maintains conversation history
    - Calculates real-time lead scores
    - Adapts responses based on lead score
    - Recommends products intelligently
    """

    def __init__(
        self,
        use_scoring: bool = True,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """
        Initialize sales agent.

        Args:
            use_scoring: Whether to use ML lead scoring
            model_name: LLM model name (defaults to config)
            temperature: LLM temperature (defaults to config)
            max_tokens: Max tokens for response (defaults to config)
        """
        self.use_scoring = use_scoring
        self.model_name = model_name or settings.llm_model
        self.temperature = temperature or settings.llm_temperature
        self.max_tokens = max_tokens or settings.llm_max_tokens

        # Initialize LLM client based on provider
        self.llm_client = self._initialize_llm()

        # Load ML scorer if enabled
        self.scorer = None
        if self.use_scoring:
            try:
                self.scorer = LeadScorer()
                if self.scorer.is_loaded:
                    logger.info("Lead scorer loaded successfully")
                else:
                    logger.warning("Lead scorer not available, using default scores")
                    self.use_scoring = False
            except Exception as e:
                logger.warning(f"Could not load lead scorer: {e}")
                self.use_scoring = False

        # Load products
        self.products = load_products()

        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor()

        # Conversation state
        self.conversation_history: List[Dict[str, str]] = []
        self.conversation_id: Optional[str] = None
        self.current_lead_score: float = 0.5  # Default neutral score

    def _initialize_llm(self):
        """Initialize LLM client based on provider setting."""
        provider = settings.llm_provider.lower()

        if provider == "anthropic":
            try:
                from anthropic import Anthropic
                if not settings.anthropic_api_key:
                    raise ValueError("ANTHROPIC_API_KEY not set")
                client = Anthropic(api_key=settings.anthropic_api_key)
                logger.info(f"Initialized Anthropic client with model {self.model_name}")
                return client
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic: {e}")
                raise

        elif provider == "openai":
            try:
                from openai import OpenAI
                if not settings.openai_api_key:
                    raise ValueError("OPENAI_API_KEY not set")
                client = OpenAI(api_key=settings.openai_api_key)
                logger.info(f"Initialized OpenAI client with model {self.model_name}")
                return client
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI: {e}")
                raise

        elif provider == "ollama":
            try:
                from openai import OpenAI
                # Ollama uses OpenAI-compatible API
                client = OpenAI(
                    base_url=settings.ollama_base_url,
                    api_key="ollama"  # Ollama doesn't require real key
                )
                logger.info(f"Initialized Ollama client with model {self.model_name}")
                return client
            except Exception as e:
                logger.error(f"Failed to initialize Ollama: {e}")
                raise

        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    @timed
    def _generate_response(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate response using LLM.

        Args:
            system_prompt: System prompt with context
            user_prompt: User's message

        Returns:
            Generated response
        """
        provider = settings.llm_provider.lower()

        try:
            if provider == "anthropic":
                response = self.llm_client.messages.create(
                    model=self.model_name,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}]
                )
                return response.content[0].text

            else:  # OpenAI or Ollama
                response = self.llm_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                return response.choices[0].message.content

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return "Desculpe, tive um problema ao processar sua mensagem. Pode tentar novamente?"

    def _calculate_lead_score(self) -> float:
        """
        Calculate lead score from current conversation.

        Returns:
            Lead score (0-1)
        """
        if not self.use_scoring or not self.scorer or not self.conversation_history:
            return 0.5  # Default neutral score

        try:
            # Create a mock conversation object from history
            mock_conversation = {
                "conversation_id": self.conversation_id or "temp",
                "customer_name": "Customer",
                "duration_seconds": int(time.time()),  # Mock duration
                "messages": [
                    {"role": msg["role"], "text": msg["text"], "timestamp": i * 10}
                    for i, msg in enumerate(self.conversation_history)
                ],
                "metadata": self._extract_metadata()
            }

            # Extract features
            features = self.feature_extractor.extract_from_conversation(mock_conversation)

            # Get score
            result = self.scorer.score_lead(features)
            return result["probability"]

        except Exception as e:
            logger.warning(f"Lead scoring failed: {e}")
            return 0.5

    def _extract_metadata(self) -> Dict[str, Any]:
        """Extract metadata from conversation history."""
        user_messages = [m for m in self.conversation_history if m["role"] == "user"]

        if not user_messages:
            return {
                "response_time_avg": 60.0,
                "message_length_avg": 50.0,
                "emoji_count": 0,
                "question_count": 0,
                "product_mentioned": "crédito",
                "mention_urgency": False,
                "mention_budget": False,
                "converted": False,
                "conversion_time": None,
                "engagement_score": 0.5
            }

        user_text = " ".join(m["text"] for m in user_messages)

        return {
            "response_time_avg": 30.0,  # Mock
            "message_length_avg": sum(len(m["text"]) for m in user_messages) / len(user_messages),
            "emoji_count": sum(m["text"].count(e) for m in user_messages for e in ["😊", "😄", "👍", "🚀"]),
            "question_count": sum(m["text"].count("?") for m in user_messages),
            "product_mentioned": "crédito",  # Mock - could be detected
            "mention_urgency": any(word in user_text.lower() for word in ["urgente", "rápido", "hoje"]),
            "mention_budget": any(word in user_text.lower() for word in ["quanto", "valor", "preço"]),
            "converted": False,
            "conversion_time": None,
            "engagement_score": min(len(user_messages) / 5.0, 1.0)
        }

    def process_message(
        self,
        user_message: str,
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a user message and generate a response.

        Args:
            user_message: The user's message
            conversation_id: Optional conversation ID for tracking

        Returns:
            Dict with response, lead_score, latency, etc.
        """
        start_time = time.time()

        # Set conversation ID
        if conversation_id:
            self.conversation_id = conversation_id

        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "text": user_message
        })

        # Limit history length
        max_history = settings.conversation_max_history
        if len(self.conversation_history) > max_history:
            self.conversation_history = self.conversation_history[-max_history:]

        # Calculate lead score
        if self.use_scoring:
            self.current_lead_score = self._calculate_lead_score()
        else:
            self.current_lead_score = 0.5

        # Format prompts
        system_prompt = format_system_prompt(
            lead_score=self.current_lead_score,
            products=self.products,
            conversation_history=self.conversation_history
        )

        user_prompt = format_user_prompt(user_message)

        # Generate response
        response = self._generate_response(system_prompt, user_prompt)

        # Add assistant response to history
        self.conversation_history.append({
            "role": "assistant",
            "text": response
        })

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        # Determine if we should recommend a product
        should_recommend = self.current_lead_score >= settings.lead_score_threshold

        # Get product recommendation if applicable
        product_recommendation = None
        if should_recommend:
            user_keywords = self._extract_keywords(user_message)
            recommendation_text = get_product_recommendation_prompt(
                self.current_lead_score,
                user_keywords,
                self.products
            )
            product_recommendation = recommendation_text

        logger.info(
            f"Processed message",
            extra={
                "conversation_id": self.conversation_id,
                "lead_score": round(self.current_lead_score, 3),
                "latency_ms": round(latency_ms, 2),
                "should_recommend": should_recommend
            }
        )

        return {
            "response": response,
            "lead_score": round(self.current_lead_score, 3),
            "lead_score_percentage": round(self.current_lead_score * 100, 1),
            "latency_ms": round(latency_ms, 2),
            "conversation_id": self.conversation_id,
            "should_recommend": should_recommend,
            "product_recommendation": product_recommendation,
            "message_count": len([m for m in self.conversation_history if m["role"] == "user"])
        }

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text."""
        text_lower = text.lower()
        keywords = []

        # Check for product-related keywords
        all_keywords = [
            "crédito", "empréstimo", "dinheiro", "investimento", "aplicar",
            "seguro", "proteção", "cartão", "limite", "previdência",
            "aposentadoria", "consórcio", "imóvel", "carro"
        ]

        for keyword in all_keywords:
            if keyword in text_lower:
                keywords.append(keyword)

        return keywords

    def reset_conversation(self):
        """Reset conversation state."""
        self.conversation_history = []
        self.conversation_id = None
        self.current_lead_score = 0.5
        logger.info("Conversation reset")

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of current conversation."""
        return {
            "conversation_id": self.conversation_id,
            "message_count": len(self.conversation_history),
            "user_messages": len([m for m in self.conversation_history if m["role"] == "user"]),
            "current_lead_score": round(self.current_lead_score, 3),
            "history": self.conversation_history
        }


if __name__ == "__main__":
    # Test agent
    print("Initializing Sales Agent...")

    try:
        agent = SalesAgent(use_scoring=True)

        print("\n=== Test Conversation ===")

        # Test messages
        test_messages = [
            "Olá, preciso de dinheiro urgente para pagar umas contas",
            "Quanto seria de juros para R$ 20 mil?",
            "Interessante! Como faço para contratar?"
        ]

        for msg in test_messages:
            print(f"\nUser: {msg}")
            result = agent.process_message(msg)
            print(f"Agent: {result['response']}")
            print(f"[Lead Score: {result['lead_score_percentage']:.1f}%, Latency: {result['latency_ms']:.0f}ms]")

            if result['should_recommend']:
                print(f"[RECOMMENDATION: {result['product_recommendation']}]")

        print("\n=== Conversation Summary ===")
        summary = agent.get_conversation_summary()
        print(f"Messages: {summary['message_count']}")
        print(f"Lead Score: {summary['current_lead_score']:.3f}")

    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have:")
        print("1. Set ANTHROPIC_API_KEY in .env")
        print("2. Trained the ML model (python scripts/train.py)")
