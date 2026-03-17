# FinBot Project Context

## Goal
Create a portfolio project combining ML (lead scoring) + LLM (sales agent) + API for selling financial products via WhatsApp-style conversations.

## Key Decisions
- **LLM**: Claude API via anthropic SDK (primary), with Ollama support as alternative
- **ML**: LightGBM for lead scoring (faster than sklearn RandomForest, better performance)
- **Data**: DuckDB for lightweight columnar storage (no PostgreSQL needed)
- **API**: FastAPI for REST API (modern, automatic OpenAPI docs, async support)
- **Optimization**: Optuna for hyperparameter tuning (state-of-the-art)

## Architecture Overview

### Data Flow
1. **Synthetic Data Generation**: `generator.py` creates realistic WhatsApp conversations
2. **Feature Extraction**: `extractor.py` transforms conversations → 15+ features
3. **ML Training**: `training.py` trains LGBMClassifier with Optuna optimization
4. **LLM Agent**: `agent.py` uses trained model score to enhance recommendations
5. **API**: FastAPI exposes scoring + chat endpoints for production use

### Key Components

**Data Pipeline** (`src/data/`)
- Generates 500+ synthetic conversations with realistic patterns
- Extracts behavioral, contextual, temporal, and linguistic features
- Uses DuckDB for efficient storage and querying

**ML Module** (`src/ml/`)
- LightGBM classifier for binary conversion prediction
- Optuna with 50-100 trials optimizing for AUC
- SHAP for feature importance analysis
- Target: AUC > 0.75

**LLM Agent** (`src/llm/`)
- Claude-powered sales agent with conversation memory
- Injects ML lead score into prompt context
- Recommends products when score > 0.7
- Evaluates responses with ROUGE, sentiment, relevance

**API** (`src/api/`)
- POST /api/score-lead: Get lead score from features
- POST /api/chat: Chat with agent + get recommendations
- GET /api/products: List available financial products
- GET /api/health: System health check

## File Organization

```
finbot/
├── data/                   # Data files and cache
├── models/                 # Trained model artifacts
├── logs/                   # Application logs
├── reports/                # Evaluation reports
├── src/
│   ├── data/              # Data generation & extraction
│   ├── ml/                # ML training & inference
│   ├── llm/               # LLM agent & evaluation
│   ├── api/               # FastAPI application
│   └── utils/             # Shared utilities
├── scripts/               # Executable scripts
├── tests/                 # Unit & integration tests
└── notebooks/             # Analysis notebooks
```

## Constraints
- **No External APIs**: Beyond Claude/Ollama (keep standalone)
- **Synthetic Data Only**: No real customer conversations
- **Minimal Dependencies**: Keep requirements.txt lean
- **Well Documented**: Every module has clear docstrings
- **Fast Setup**: Runnable in < 5 mins on fresh machine

## Success Criteria
- ✓ Portfolio-ready (showcaseable at interviews)
- ✓ AUC > 0.75 on lead scoring
- ✓ LLM responses relevant + fast (< 500ms p95)
- ✓ API fully functional with Swagger docs
- ✓ Complete test coverage
- ✓ Clean, maintainable code

## Environment Variables
Create `.env` file:
```
ANTHROPIC_API_KEY=sk-ant-...
OLLAMA_BASE_URL=http://localhost:11434  # Optional
MODEL_PATH=models/lead_scorer.pkl
DATA_PATH=data/
LOG_LEVEL=INFO
```

## Common Commands

**Setup**
```bash
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

**Training**
```bash
python scripts/train.py --n-trials 50 --test-size 0.2
```

**Demo**
```bash
python scripts/demo.py  # Interactive chat
```

**API**
```bash
uvicorn src.api.main:app --reload --port 8000
# Visit http://localhost:8000/docs
```

**Tests**
```bash
pytest tests/ -v --cov=src
```

**Evaluation**
```bash
python scripts/evaluate.py  # Generates reports/evaluation.html
```

## Development Notes

- **Data Generation**: Uses templates + random variations to create realistic patterns
- **Feature Engineering**: Focus on behavioral signals (response time, engagement)
- **Model Selection**: LightGBM chosen for speed + performance on tabular data
- **LLM Integration**: Score influences prompt temperature + recommendation aggressiveness
- **Evaluation**: Both traditional ML metrics + LLM-specific (ROUGE, relevance)

## Future Enhancements (Bonus)
- Docker + docker-compose for one-command deploy
- GitHub Actions CI/CD pipeline
- Real-time Plotly dashboard
- Fine-tuned embeddings for product matching
- A/B testing framework for prompt optimization
