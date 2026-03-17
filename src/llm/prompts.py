"""
Prompt templates for the LLM sales agent.
Contains system prompts and response templates.
"""

from typing import Dict, Any, List
import json
from pathlib import Path

from src.config import settings

# System prompt for the sales agent
SYSTEM_PROMPT = """Você é um assistente de vendas especializado em produtos financeiros da FinBot.

Seu objetivo é:
1. Entender a necessidade do cliente
2. Recomendar o produto financeiro mais adequado
3. Responder dúvidas de forma clara e objetiva
4. Fechar a venda de forma natural

PRODUTOS DISPONÍVEIS:
{products_info}

INFORMAÇÃO DO LEAD:
Lead Score (confiança de conversão): {lead_score}%
Nível de engajamento: {engagement_level}

DIRETRIZES DE COMPORTAMENTO:
- Se o Lead Score estiver acima de 70%, seja mais assertivo e proativo na recomendação
- Se o Lead Score estiver entre 50-70%, explore mais as necessidades antes de recomendar
- Se o Lead Score estiver abaixo de 50%, foque em educar e construir confiança
- Mantenha respostas curtas (2-3 linhas), como em uma conversa de WhatsApp
- Use linguagem natural e amigável, mas profissional
- Não use emojis em excesso (máximo 1 por mensagem, quando apropriado)
- Quando recomendar um produto, destaque 2-3 benefícios principais
- Se o cliente mostrar urgência, priorize rapidez na resposta

CONTEXTO DA CONVERSA:
{conversation_context}

Responda ao cliente de forma natural, seguindo as diretrizes acima."""


USER_PROMPT_TEMPLATE = """Cliente: {user_message}

Assistente:"""


def format_system_prompt(
    lead_score: float,
    products: List[Dict[str, Any]],
    conversation_history: List[Dict[str, str]] = None
) -> str:
    """
    Format the system prompt with dynamic context.

    Args:
        lead_score: Lead score (0-1, will be converted to percentage)
        products: List of product dicts
        conversation_history: Optional conversation history

    Returns:
        Formatted system prompt
    """
    # Convert score to percentage
    score_pct = lead_score * 100

    # Determine engagement level
    if score_pct >= 70:
        engagement_level = "ALTO - Cliente muito engajado e propenso a converter"
    elif score_pct >= 50:
        engagement_level = "MÉDIO - Cliente interessado, mas ainda avaliando"
    else:
        engagement_level = "BAIXO - Cliente em fase inicial de descoberta"

    # Format products information
    products_lines = []
    for product in products:
        line = f"- {product['name']}: {product['description']}"
        if 'benefits' in product:
            benefits = ", ".join(product['benefits'][:2])  # First 2 benefits
            line += f" ({benefits})"
        products_lines.append(line)

    products_info = "\n".join(products_lines)

    # Format conversation context
    if conversation_history:
        context_lines = []
        for msg in conversation_history[-5:]:  # Last 5 messages
            role = "Cliente" if msg["role"] == "user" else "Você"
            context_lines.append(f"{role}: {msg['text']}")
        conversation_context = "\n".join(context_lines)
    else:
        conversation_context = "Esta é a primeira mensagem da conversa."

    return SYSTEM_PROMPT.format(
        lead_score=f"{score_pct:.1f}",
        engagement_level=engagement_level,
        products_info=products_info,
        conversation_context=conversation_context
    )


def format_user_prompt(user_message: str) -> str:
    """
    Format user message into prompt.

    Args:
        user_message: The user's message

    Returns:
        Formatted prompt
    """
    return USER_PROMPT_TEMPLATE.format(user_message=user_message)


def load_products(products_path: Path = None) -> List[Dict[str, Any]]:
    """
    Load products from JSON file.

    Args:
        products_path: Path to products.json

    Returns:
        List of product dicts
    """
    if products_path is None:
        products_path = settings.products_path

    if not products_path.exists():
        raise FileNotFoundError(f"Products file not found: {products_path}")

    with open(products_path, 'r', encoding='utf-8') as f:
        products = json.load(f)

    return products


def get_product_recommendation_prompt(
    lead_score: float,
    user_interest_keywords: List[str],
    products: List[Dict[str, Any]]
) -> str:
    """
    Generate a prompt for product recommendation based on interest keywords.

    Args:
        lead_score: Lead score (0-1)
        user_interest_keywords: Keywords from user's messages
        products: List of products

    Returns:
        Recommendation prompt
    """
    # Find matching products
    matching_products = []
    for product in products:
        # Check if product's min_score is met
        if lead_score >= product.get('min_score', 0.5):
            # Check keyword overlap
            product_keywords = set(product.get('keywords', []))
            user_keywords = set(user_interest_keywords)
            overlap = product_keywords.intersection(user_keywords)

            if overlap or not user_interest_keywords:
                matching_products.append({
                    'product': product,
                    'relevance': len(overlap)
                })

    # Sort by relevance
    matching_products = sorted(matching_products, key=lambda x: x['relevance'], reverse=True)

    if not matching_products:
        return "Nenhum produto específico identificado ainda. Continue explorando as necessidades do cliente."

    # Recommend top product
    top_product = matching_products[0]['product']

    recommendation = f"RECOMENDAÇÃO PRIORITÁRIA: {top_product['name']}\n"
    recommendation += f"Motivo: {top_product['description']}\n"
    recommendation += f"Destaque estes benefícios: {', '.join(top_product['benefits'][:3])}"

    return recommendation


# Evaluation prompts for testing
EVALUATION_PROMPT = """Você é um avaliador de qualidade de respostas de chatbots de vendas.

Avalie a seguinte resposta do assistente com base nestes critérios:
1. Relevância: A resposta aborda a pergunta/necessidade do cliente? (0-5)
2. Claridade: A resposta é clara e fácil de entender? (0-5)
3. Persuasão: A resposta é convincente e incentiva a conversão? (0-5)
4. Naturalidade: A resposta parece natural em uma conversa de WhatsApp? (0-5)

CONTEXTO:
{context}

MENSAGEM DO CLIENTE:
{user_message}

RESPOSTA DO ASSISTENTE:
{assistant_response}

Responda APENAS em formato JSON:
{{
    "relevancia": <score 0-5>,
    "claridade": <score 0-5>,
    "persuasao": <score 0-5>,
    "naturalidade": <score 0-5>,
    "score_total": <soma dos scores>,
    "justificativa": "<breve explicação>"
}}"""


def format_evaluation_prompt(
    context: str,
    user_message: str,
    assistant_response: str
) -> str:
    """
    Format evaluation prompt for LLM-based response evaluation.

    Args:
        context: Conversation context
        user_message: User's message
        assistant_response: Assistant's response to evaluate

    Returns:
        Formatted evaluation prompt
    """
    return EVALUATION_PROMPT.format(
        context=context,
        user_message=user_message,
        assistant_response=assistant_response
    )


if __name__ == "__main__":
    # Test prompt formatting
    products = load_products()

    # Test system prompt
    system_prompt = format_system_prompt(
        lead_score=0.75,
        products=products,
        conversation_history=[
            {"role": "user", "text": "Oi, preciso de dinheiro urgente"},
            {"role": "assistant", "text": "Olá! Posso te ajudar com isso. Quanto você precisa?"}
        ]
    )

    print("=== SYSTEM PROMPT ===")
    print(system_prompt)

    print("\n=== USER PROMPT ===")
    user_prompt = format_user_prompt("Preciso de R$ 20 mil, pode ser?")
    print(user_prompt)

    print("\n=== RECOMMENDATION ===")
    recommendation = get_product_recommendation_prompt(
        lead_score=0.8,
        user_interest_keywords=["crédito", "urgente", "dinheiro"],
        products=products
    )
    print(recommendation)
