"""
Generate synthetic WhatsApp-style conversations for training.
Creates realistic conversation patterns with varying engagement levels.
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from src.config import settings, DATA_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Customer names for variety
CUSTOMER_NAMES = [
    "João Silva", "Maria Santos", "Pedro Costa", "Ana Oliveira", "Carlos Souza",
    "Juliana Lima", "Rafael Mendes", "Fernanda Alves", "Lucas Rodrigues", "Camila Ferreira",
    "Bruno Carvalho", "Patricia Ribeiro", "Marcos Pereira", "Beatriz Gomes", "Felipe Martins",
    "Amanda Rocha", "Thiago Barbosa", "Larissa Dias", "Gabriel Araújo", "Mariana Castro",
    "Ricardo Cavalcante", "Natália Moreira", "Rodrigo Cardoso", "Vanessa Correia", "Diego Monteiro",
    "Isabela Teixeira", "Leonardo Pinto", "Carolina Freitas", "Eduardo Nascimento", "Letícia Azevedo"
]

# Product-related keywords
PRODUCT_KEYWORDS = {
    "crédito": ["empréstimo", "dinheiro", "urgente", "precisando", "boleto", "conta", "dívida"],
    "investimento": ["aplicar", "rentabilidade", "poupar", "futuro", "guardar dinheiro", "ações", "renda"],
    "seguro": ["proteção", "família", "vida", "saúde", "carro", "casa", "acidentes"],
    "cartão": ["limite", "compras", "cashback", "anuidade", "parcelado"],
    "previdência": ["aposentadoria", "longo prazo", "futuro", "PGBL", "VGBL"],
    "consórcio": ["imóvel", "carro", "parcela", "contemplação"]
}

# Conversation templates
BOT_GREETINGS = [
    "Olá {name}! Tudo bem? Sou o assistente financeiro da FinBot. Como posso te ajudar hoje?",
    "Oi {name}! Bem-vindo(a) à FinBot! Estou aqui para ajudar com soluções financeiras. O que você procura?",
    "Olá {name}! Prazer em conversar com você! Tem alguma necessidade financeira que eu possa ajudar?",
]

USER_INTERESTED_RESPONSES = [
    "Sim, me interessei", "Quero saber mais", "Pode me explicar melhor?",
    "Interessante, como funciona?", "Me conte mais sobre isso",
    "Gostei! Quais são as condições?", "Opa, me diz mais"
]

USER_NOT_INTERESTED = [
    "Não obrigado", "Agora não", "Não tenho interesse",
    "Depois eu vejo", "Não é pra mim", "Vou pensar"
]

USER_QUESTIONS = [
    "Qual é a taxa de juros?", "Quanto custa?", "Qual o prazo?",
    "Preciso de garantia?", "Como faço para contratar?", "Vocês consultam o SPC?"
]

USER_URGENCY = [
    "Preciso urgente!", "É para hoje!", "Quando consigo?",
    "Quanto tempo demora?", "Pode ser rápido?"
]

BOT_PRODUCT_PITCH = {
    "crédito": [
        "Temos o Crédito Flow! Até R$ 50 mil com aprovação em até 24h. Juros a partir de 1,99% ao mês.",
        "Nosso empréstimo pessoal é ideal pra você! Até R$ 50k, sem consulta ao SPC/Serasa.",
    ],
    "investimento": [
        "Conheça o InvestBro! Plataforma de investimentos com taxa zero e acesso a +500 ativos.",
        "Que tal investir com taxa zero? Temos fundos, ações e renda fixa com assessoria gratuita!",
    ],
    "seguro": [
        "O SeguroMax protege você e sua família! Cobertura até R$ 500k e telemedicina 24h por R$ 29,90/mês.",
        "Proteção completa para sua família! Seguro de vida com assistência funeral e telemedicina.",
    ],
    "cartão": [
        "O Cartão Pro é sem anuidade e dá até 3% de cashback! Limite inicial de até R$ 10k.",
        "Que tal um cartão sem anuidade para sempre? Até 3% de cashback em todas as compras!",
    ],
    "previdência": [
        "A Previd Futuro é um plano de previdência privada com benefícios fiscais e gestão profissional.",
        "Pense no seu futuro! Previdência com rentabilidade acima da inflação e portabilidade grátis.",
    ],
    "consórcio": [
        "Consórcio Fácil! Realize o sonho da casa ou carro próprio sem juros, em até 120x.",
        "Sem entrada e sem juros! Consórcio de imóveis e veículos com parcelas que cabem no bolso.",
    ]
}


def generate_conversation(
    customer_name: str,
    product_category: str,
    will_convert: bool,
    engagement_level: float
) -> dict[str, Any]:
    """
    Generate a single synthetic conversation.

    Args:
        customer_name: Name of the customer
        product_category: Type of product (crédito, investimento, etc)
        will_convert: Whether customer will convert
        engagement_level: 0-1, higher means more engaged

    Returns:
        Conversation dict with messages and metadata
    """
    conversation_id = f"conv_{random.randint(10000, 99999)}"
    messages = []
    timestamp = 0

    # Bot greeting
    greeting = random.choice(BOT_GREETINGS).format(name=customer_name.split()[0])
    messages.append({"role": "bot", "text": greeting, "timestamp": timestamp})

    # User initial response (based on engagement)
    response_time = random.uniform(1, 30) if engagement_level > 0.5 else random.uniform(30, 120)
    timestamp += response_time

    # User shows interest based on product keywords
    interest_keywords = random.sample(PRODUCT_KEYWORDS[product_category], k=min(2, len(PRODUCT_KEYWORDS[product_category])))
    user_intro = f"Olá! Estou procurando {random.choice(interest_keywords)}"

    if random.random() < engagement_level:
        user_intro += f", você pode me ajudar? {'😊' if random.random() > 0.5 else ''}"

    messages.append({"role": "user", "text": user_intro, "timestamp": timestamp})

    # Bot pitches product
    timestamp += random.uniform(2, 5)
    pitch = random.choice(BOT_PRODUCT_PITCH[product_category])
    messages.append({"role": "bot", "text": pitch, "timestamp": timestamp})

    # Conversation continues based on engagement
    num_turns = random.randint(3, 8) if engagement_level > 0.6 else random.randint(2, 4)
    response_times = []
    emoji_count = 0
    question_count = 0

    for turn in range(num_turns):
        # User response time (faster if more engaged)
        if engagement_level > 0.7:
            response_time = random.uniform(1, 10)
        elif engagement_level > 0.4:
            response_time = random.uniform(10, 40)
        else:
            response_time = random.uniform(40, 180)

        response_times.append(response_time)
        timestamp += response_time

        # User message (varies by engagement and conversion intent)
        if will_convert and turn >= num_turns - 2:
            # About to convert - show strong interest
            user_msg = random.choice(USER_INTERESTED_RESPONSES + USER_URGENCY)
            if random.random() > 0.6:
                user_msg += " 😄"
                emoji_count += 1
        elif engagement_level > 0.6:
            # High engagement - ask questions
            user_msg = random.choice(USER_QUESTIONS + USER_INTERESTED_RESPONSES)
            if "?" in user_msg:
                question_count += 1
        else:
            # Low engagement
            user_msg = random.choice(USER_NOT_INTERESTED + ["Ok", "Entendi", "Hmm"])

        messages.append({"role": "user", "text": user_msg, "timestamp": timestamp})

        # Bot response
        timestamp += random.uniform(2, 8)

        if will_convert and turn >= num_turns - 1:
            bot_msg = "Perfeito! Vou te enviar o link para finalizar. É super rápido! 🚀"
        elif "quanto" in user_msg.lower() or "taxa" in user_msg.lower():
            bot_msg = "As condições são super competitivas! Posso fazer uma simulação personalizada para você."
        elif any(word in user_msg.lower() for word in ["não", "depois", "agora não"]):
            bot_msg = "Sem problema! Quando quiser, é só me chamar. Estou sempre aqui! 😊"
        else:
            bot_msg = "Com certeza! Nosso produto tem várias vantagens e é bem fácil de contratar."

        messages.append({"role": "bot", "text": bot_msg, "timestamp": timestamp})

    # Calculate metadata
    total_duration = timestamp
    avg_response_time = sum(response_times) / len(response_times) if response_times else 60
    user_messages = [m for m in messages if m["role"] == "user"]
    avg_message_length = sum(len(m["text"]) for m in user_messages) / len(user_messages)

    # Conversion time (when they decided to convert)
    conversion_time = int(timestamp * random.uniform(0.6, 0.9)) if will_convert else None

    metadata = {
        "response_time_avg": round(avg_response_time, 2),
        "message_length_avg": round(avg_message_length, 2),
        "emoji_count": emoji_count,
        "question_count": question_count,
        "product_mentioned": product_category,
        "mention_urgency": any(word in " ".join(m["text"] for m in user_messages).lower()
                              for word in ["urgente", "rápido", "hoje", "quando"]),
        "mention_budget": any(word in " ".join(m["text"] for m in user_messages).lower()
                             for word in ["quanto", "valor", "preço", "taxa", "custo"]),
        "converted": will_convert,
        "conversion_time": conversion_time,
        "engagement_score": round(engagement_level, 2)
    }

    return {
        "conversation_id": conversation_id,
        "customer_name": customer_name,
        "duration_seconds": int(total_duration),
        "messages": messages,
        "metadata": metadata
    }


def generate_dataset(
    n_conversations: int = 500,
    conversion_rate: float = 0.3,
    output_path: Optional[Path] = None
) -> list[dict[str, Any]]:
    """
    Generate a complete dataset of synthetic conversations.

    Args:
        n_conversations: Number of conversations to generate
        conversion_rate: Proportion that will convert (0-1)
        output_path: Path to save JSON file (optional)

    Returns:
        List of conversation dicts
    """
    logger.info(f"Generating {n_conversations} synthetic conversations...")

    conversations = []
    products = list(PRODUCT_KEYWORDS.keys())

    for i in range(n_conversations):
        # Determine conversion
        will_convert = random.random() < conversion_rate

        # Engagement correlates with conversion (but with noise)
        if will_convert:
            engagement = random.uniform(0.6, 1.0)
        else:
            engagement = random.uniform(0.0, 0.7)

        # Random customer and product
        customer_name = random.choice(CUSTOMER_NAMES)
        product = random.choice(products)

        conv = generate_conversation(
            customer_name=customer_name,
            product_category=product,
            will_convert=will_convert,
            engagement_level=engagement
        )

        conversations.append(conv)

        if (i + 1) % 100 == 0:
            logger.info(f"Generated {i + 1}/{n_conversations} conversations")

    # Save to file if path provided
    if output_path is None:
        output_path = settings.synthetic_conversations_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(conversations)} conversations to {output_path}")

    # Log statistics
    conversions = sum(1 for c in conversations if c["metadata"]["converted"])
    logger.info(f"Dataset stats: {conversions}/{len(conversations)} converted ({conversions/len(conversations)*100:.1f}%)")

    return conversations


if __name__ == "__main__":
    # Generate dataset
    conversations = generate_dataset(n_conversations=500, conversion_rate=0.3)

    # Print sample
    print("\n=== Sample Conversation ===")
    sample = conversations[0]
    print(f"Customer: {sample['customer_name']}")
    print(f"Duration: {sample['duration_seconds']}s")
    print(f"Converted: {sample['metadata']['converted']}")
    print("\nMessages:")
    for msg in sample['messages'][:6]:
        print(f"  [{msg['role']}] {msg['text']}")
