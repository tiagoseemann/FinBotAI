"""
Pydantic models for API request/response schemas.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, field_validator


# Request models
class LeadScoreRequest(BaseModel):
    """Request for lead scoring endpoint."""

    features: Dict[str, float] = Field(
        ...,
        description="Dictionary of feature names to values",
        examples=[{
            "avg_response_time": 15.0,
            "message_count": 5,
            "emoji_count": 2,
            "engagement_score": 0.75
        }]
    )

    @field_validator('features')
    @classmethod
    def validate_features(cls, v):
        if not v:
            raise ValueError("Features dictionary cannot be empty")
        return v


class ChatRequest(BaseModel):
    """Request for chat endpoint."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User's message",
        examples=["Olá, preciso de um empréstimo urgente"]
    )

    conversation_id: Optional[str] = Field(
        None,
        description="Optional conversation ID for continuity",
        examples=["conv_12345"]
    )

    use_scoring: bool = Field(
        True,
        description="Whether to use ML lead scoring"
    )

    @field_validator('message')
    @classmethod
    def validate_message(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Message cannot be empty or only whitespace")
        return v


# Response models
class LeadScoreResponse(BaseModel):
    """Response from lead scoring endpoint."""

    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Lead score probability (0-1)"
    )

    score_percentage: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Lead score as percentage"
    )

    prediction: int = Field(
        ...,
        ge=0,
        le=1,
        description="Binary prediction (0=not convert, 1=convert)"
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in prediction"
    )

    top_features: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Top contributing features"
    )


class ChatResponse(BaseModel):
    """Response from chat endpoint."""

    response: str = Field(
        ...,
        description="Agent's response message"
    )

    lead_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Current lead score"
    )

    lead_score_percentage: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Lead score as percentage"
    )

    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Response latency in milliseconds"
    )

    conversation_id: Optional[str] = Field(
        None,
        description="Conversation ID"
    )

    should_recommend: bool = Field(
        ...,
        description="Whether product recommendation should be made"
    )

    product_recommendation: Optional[str] = Field(
        None,
        description="Product recommendation text if applicable"
    )

    message_count: int = Field(
        ...,
        ge=0,
        description="Number of user messages in conversation"
    )


class Product(BaseModel):
    """Product information."""

    id: str = Field(..., description="Product ID")
    name: str = Field(..., description="Product name")
    description: str = Field(..., description="Product description")
    keywords: List[str] = Field(default_factory=list, description="Product keywords")
    min_score: float = Field(..., ge=0.0, le=1.0, description="Minimum lead score for recommendation")
    benefits: List[str] = Field(default_factory=list, description="Product benefits")


class ProductsResponse(BaseModel):
    """Response from products endpoint."""

    products: List[Product] = Field(..., description="List of available products")
    total: int = Field(..., ge=0, description="Total number of products")


class HealthResponse(BaseModel):
    """Response from health check endpoint."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether ML model is loaded")
    llm_provider: str = Field(..., description="LLM provider in use")
    llm_model: str = Field(..., description="LLM model name")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    status_code: int = Field(..., description="HTTP status code")


# Conversation models
class ConversationMessage(BaseModel):
    """A single message in a conversation."""

    role: str = Field(..., description="Message role (user or assistant)")
    text: str = Field(..., description="Message text")
    timestamp: Optional[float] = Field(None, description="Message timestamp")


class ConversationSummary(BaseModel):
    """Summary of a conversation."""

    conversation_id: Optional[str] = Field(None, description="Conversation ID")
    message_count: int = Field(..., ge=0, description="Total messages")
    user_messages: int = Field(..., ge=0, description="User messages count")
    current_lead_score: float = Field(..., ge=0.0, le=1.0, description="Current lead score")
    history: List[ConversationMessage] = Field(default_factory=list, description="Message history")


# Evaluation models
class EvaluationRequest(BaseModel):
    """Request for evaluation endpoint."""

    conversations: List[Dict[str, Any]] = Field(
        ...,
        min_length=1,
        description="Test conversations to evaluate"
    )

    n_trials: int = Field(
        20,
        ge=1,
        le=100,
        description="Number of test conversations"
    )


class EvaluationResponse(BaseModel):
    """Response from evaluation endpoint."""

    total_conversations: int = Field(..., ge=0)
    total_messages: int = Field(..., ge=0)
    latency_mean_ms: float = Field(..., ge=0.0)
    latency_p95_ms: float = Field(..., ge=0.0)
    lead_score_mean: float = Field(..., ge=0.0, le=1.0)
    rouge1_mean: float = Field(..., ge=0.0, le=1.0)
    rouge2_mean: float = Field(..., ge=0.0, le=1.0)
    rougeL_mean: float = Field(..., ge=0.0, le=1.0)


if __name__ == "__main__":
    # Test models
    from pydantic import ValidationError

    # Test valid request
    try:
        req = ChatRequest(
            message="Olá, preciso de ajuda",
            conversation_id="conv_123"
        )
        print("✓ Valid ChatRequest:", req.model_dump())
    except ValidationError as e:
        print("✗ Error:", e)

    # Test invalid request (empty message)
    try:
        req = ChatRequest(message="   ")
        print("✗ Should have failed!")
    except ValidationError as e:
        print("✓ Correctly rejected empty message")

    # Test response
    response = ChatResponse(
        response="Olá! Como posso ajudar?",
        lead_score=0.75,
        lead_score_percentage=75.0,
        latency_ms=234.5,
        should_recommend=True,
        message_count=1
    )
    print("\n✓ ChatResponse:", response.model_dump_json(indent=2))
