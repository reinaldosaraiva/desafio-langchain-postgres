#!/usr/bin/env python3
"""
CLI de Chat RAG para interação com usuário

Uso:
  python src/chat.py
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "default")
SEARCH_K = int(os.getenv("SEARCH_K", "10"))
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash-lite")  # Modelo obrigatório

POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "postgres")

CONNECTION_STRING = (
    f"postgresql+psycopg://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

# Prompt OBRIGATÓRIO conforme requisitos
RAG_PROMPT_TEMPLATE = """CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def get_llm():
    """Retorna instância de LLM configurada (modelo obrigatório)."""
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        raise ValueError("GOOGLE_API_KEY não definida")
    
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,  # gemini-2.5-flash-lite (obrigatório)
        google_api_key=api_key,
        temperature=1.0,
        max_retries=3,
        timeout=60,
    )


def get_embeddings():
    """Retorna instância de embeddings configurada."""
    api_key = os.getenv("GOOGLE_API_KEY")
    
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key,
        task_type="RETRIEVAL_QUERY",
    )


def get_vectorstore():
    """Retorna instância do vectorstore configurado."""
    embeddings = get_embeddings()
    
    return PGVector(
        embeddings=embeddings,
        connection=CONNECTION_STRING,
        collection_name=COLLECTION_NAME,
        use_jsonb=True,
    )


def format_context(results):
    """Formata resultados para o prompt (resultados concatenados)."""
    contexts = []
    
    for i, (doc, score) in enumerate(results, 1):
        content = doc.page_content
        context = f"[CONTEXTO {i}]\n{content}"
        contexts.append(context)
    
    return "\n\n".join(contexts)


def ask_question(pergunta: str):
    """
    Processa pergunta do usuário.
    
    Passos:
    1. Vetorizar a pergunta
    2. Buscar os 10 resultados mais relevantes (k=10)
    3. Montar o prompt e chamar a LLM
    4. Retornar a resposta ao usuário
    """
    vectorstore = get_vectorstore()
    
    # 1. Vetorizar e buscar (Requisito: k=10)
    results = vectorstore.similarity_search_with_score(pergunta, k=SEARCH_K)
    
    if not results:
        return "Não tenho informações necessárias para responder sua pergunta.", results
    
    # 2. Montar o prompt
    contexto = format_context(results)
    prompt = RAG_PROMPT_TEMPLATE.format(contexto=contexto, pergunta=pergunta)
    
    # 3. Chamar a LLM
    llm = get_llm()
    resposta = llm.invoke(prompt)
    
    return resposta.content, results


def main():
    """Loop principal do chat."""
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║              CLI DE BUSCA SEMÂNTICA - RAG                     ║")
    print("║              LangChain + PostgreSQL + pgVector                 ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    print("💡 Digite sua pergunta abaixo:")
    print("⚠️  Pressione Ctrl+C para sair")
    print()
    
    try:
        while True:
            pergunta = input("PERGUNTA: ").strip()
            
            if not pergunta:
                continue
            
            if pergunta.lower() in ['sair', 'exit', 'quit']:
                break
            
            # Processar pergunta
            resposta, results = ask_question(pergunta)
            
            # Exibir resposta
            print(f"RESPOSTA: {resposta}")
            print()
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Saindo...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERRO: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
