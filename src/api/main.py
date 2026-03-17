"""
FastAPI application for FinBot.
Main entry point for the REST API.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from src.api.routes import router, initialize_routes
from src.api.models import ErrorResponse
from src.config import settings, validate_settings
from src.ml.model import LeadScorer
from src.llm.agent import SalesAgent
from src.utils.logger import get_logger
from src import __version__

logger = get_logger(__name__)


# Global state
scorer: LeadScorer = None
agent: SalesAgent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("=" * 60)
    logger.info("Starting FinBot API")
    logger.info("=" * 60)

    # Validate settings
    errors = validate_settings()
    if errors:
        logger.warning("Configuration issues detected:")
        for error in errors:
            logger.warning(f"  - {error}")

    # Load ML model
    global scorer, agent
    try:
        logger.info("Loading ML model...")
        scorer = LeadScorer()
        if scorer.is_loaded:
            logger.info("✓ ML model loaded successfully")
        else:
            logger.warning("✗ ML model not found - scorer will not be available")
    except Exception as e:
        logger.error(f"Failed to load ML model: {e}")

    # Initialize LLM agent (template)
    try:
        logger.info("Initializing LLM agent...")
        agent = SalesAgent(use_scoring=scorer is not None and scorer.is_loaded)
        logger.info(f"✓ LLM agent initialized with {settings.llm_provider}/{settings.llm_model}")
    except Exception as e:
        logger.error(f"Failed to initialize LLM agent: {e}")
        agent = None

    # Initialize routes with dependencies
    initialize_routes(scorer, agent)

    logger.info(f"API Version: {__version__}")
    logger.info(f"Host: {settings.api_host}:{settings.api_port}")
    logger.info("=" * 60)

    yield

    # Shutdown
    logger.info("Shutting down FinBot API")


# Create FastAPI app
app = FastAPI(
    title="FinBot API",
    description="AI Sales Agent for Financial Products with ML-powered Lead Scoring",
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation error",
            "detail": str(exc),
            "status_code": 422
        }
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle value errors."""
    logger.warning(f"Value error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "Invalid value",
            "detail": str(exc),
            "status_code": 400
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred",
            "status_code": 500
        }
    )


# Include routers
app.include_router(router, prefix="/api")


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "FinBot API",
        "version": __version__,
        "description": "AI Sales Agent for Financial Products",
        "docs": "/docs",
        "health": "/api/health",
        "endpoints": {
            "score_lead": "POST /api/score-lead",
            "chat": "POST /api/chat",
            "products": "GET /api/products",
            "health": "GET /api/health"
        }
    }


# Middleware for logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests."""
    logger.info(
        f"Request: {request.method} {request.url.path}",
        extra={
            "method": request.method,
            "path": request.url.path,
            "client": request.client.host if request.client else None
        }
    )

    response = await call_next(request)

    logger.info(
        f"Response: {response.status_code}",
        extra={
            "status_code": response.status_code,
            "path": request.url.path
        }
    )

    return response


if __name__ == "__main__":
    import uvicorn

    # Run with uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower()
    )
