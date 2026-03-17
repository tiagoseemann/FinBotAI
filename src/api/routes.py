"""
API route handlers for FinBot.
"""

from typing import Dict
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse

from src.api.models import (
    LeadScoreRequest,
    LeadScoreResponse,
    ChatRequest,
    ChatResponse,
    ProductsResponse,
    Product,
    HealthResponse,
    ErrorResponse,
    ConversationSummary
)
from src.config import settings
from src.ml.model import LeadScorer
from src.llm.agent import SalesAgent
from src.llm.prompts import load_products
from src.utils.logger import get_logger
from src import __version__

logger = get_logger(__name__)

# Create router
router = APIRouter()

# Global instances (will be initialized in main.py)
_scorer: LeadScorer = None
_agent: SalesAgent = None
_agents: Dict[str, SalesAgent] = {}  # Conversation ID -> Agent


def get_scorer() -> LeadScorer:
    """Dependency to get scorer instance."""
    if _scorer is None:
        raise HTTPException(status_code=503, detail="ML model not loaded")
    return _scorer


def get_or_create_agent(conversation_id: str = None) -> SalesAgent:
    """Get existing agent or create new one for conversation."""
    if conversation_id and conversation_id in _agents:
        return _agents[conversation_id]

    # Create new agent
    agent = SalesAgent(use_scoring=True)

    if conversation_id:
        agent.conversation_id = conversation_id
        _agents[conversation_id] = agent

    return agent


@router.post(
    "/score-lead",
    response_model=LeadScoreResponse,
    summary="Score a lead",
    description="Calculate lead score from conversation features"
)
async def score_lead(
    request: LeadScoreRequest,
    scorer: LeadScorer = Depends(get_scorer)
):
    """
    Score a lead based on extracted features.

    Returns probability of conversion, prediction, and top contributing features.
    """
    try:
        logger.info("Scoring lead", extra={"n_features": len(request.features)})

        result = scorer.score_lead(request.features)

        return LeadScoreResponse(
            score=result["probability"],
            score_percentage=result["score_percentage"],
            prediction=result["prediction"],
            confidence=result["confidence"],
            top_features=result["top_features"]
        )

    except Exception as e:
        logger.error(f"Lead scoring failed: {e}")
        raise HTTPException(status_code=500, detail=f"Lead scoring failed: {str(e)}")


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat with sales agent",
    description="Send a message to the AI sales agent and get a response"
)
async def chat(request: ChatRequest):
    """
    Chat with the AI sales agent.

    The agent will:
    - Maintain conversation history
    - Calculate lead score in real-time
    - Recommend products when appropriate
    - Adapt responses based on engagement
    """
    try:
        logger.info(
            "Processing chat message",
            extra={
                "conversation_id": request.conversation_id,
                "message_length": len(request.message)
            }
        )

        # Get or create agent for this conversation
        agent = get_or_create_agent(request.conversation_id)

        # Process message
        result = agent.process_message(
            user_message=request.message,
            conversation_id=request.conversation_id
        )

        # Update agent storage with new conversation ID if created
        if result["conversation_id"] and result["conversation_id"] not in _agents:
            _agents[result["conversation_id"]] = agent

        return ChatResponse(**result)

    except Exception as e:
        logger.error(f"Chat processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


@router.get(
    "/products",
    response_model=ProductsResponse,
    summary="List products",
    description="Get list of all available financial products"
)
async def get_products():
    """
    Get list of available financial products.

    Returns product details including benefits, keywords, and minimum lead score.
    """
    try:
        products_data = load_products()

        products = [Product(**p) for p in products_data]

        return ProductsResponse(
            products=products,
            total=len(products)
        )

    except Exception as e:
        logger.error(f"Failed to load products: {e}")
        raise HTTPException(status_code=500, detail="Failed to load products")


@router.get(
    "/conversation/{conversation_id}",
    response_model=ConversationSummary,
    summary="Get conversation",
    description="Get summary and history of a conversation"
)
async def get_conversation(conversation_id: str):
    """
    Get conversation summary and history.
    """
    if conversation_id not in _agents:
        raise HTTPException(status_code=404, detail="Conversation not found")

    agent = _agents[conversation_id]
    summary = agent.get_conversation_summary()

    return ConversationSummary(**summary)


@router.delete(
    "/conversation/{conversation_id}",
    summary="Delete conversation",
    description="Delete a conversation and reset its state"
)
async def delete_conversation(conversation_id: str):
    """
    Delete a conversation and free up resources.
    """
    if conversation_id in _agents:
        _agents[conversation_id].reset_conversation()
        del _agents[conversation_id]
        return {"message": "Conversation deleted", "conversation_id": conversation_id}

    raise HTTPException(status_code=404, detail="Conversation not found")


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check API health and status"
)
async def health_check():
    """
    Health check endpoint.

    Returns service status and configuration information.
    """
    model_loaded = _scorer is not None and _scorer.is_loaded

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        version=__version__,
        model_loaded=model_loaded,
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model
    )


@router.get(
    "/stats",
    summary="Get statistics",
    description="Get API usage statistics"
)
async def get_stats():
    """
    Get API statistics.
    """
    return {
        "active_conversations": len(_agents),
        "conversation_ids": list(_agents.keys())[:10],  # First 10
        "model_loaded": _scorer is not None and _scorer.is_loaded,
        "total_conversations_tracked": len(_agents)
    }


# Initialize function to be called from main.py
def initialize_routes(scorer: LeadScorer, agent: SalesAgent):
    """
    Initialize route dependencies.

    Args:
        scorer: Loaded LeadScorer instance
        agent: Default SalesAgent instance (template)
    """
    global _scorer, _agent
    _scorer = scorer
    _agent = agent
    logger.info("API routes initialized")


if __name__ == "__main__":
    # Test route definitions
    print("API Routes:")
    print(f"  POST /score-lead - Score a lead")
    print(f"  POST /chat - Chat with agent")
    print(f"  GET  /products - List products")
    print(f"  GET  /conversation/{{id}} - Get conversation")
    print(f"  DELETE /conversation/{{id}} - Delete conversation")
    print(f"  GET  /health - Health check")
    print(f"  GET  /stats - Usage statistics")
