#!/usr/bin/env python3
"""
Script de busca semântica

Uso:
  python src/search.py "Sua pergunta"
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "default")
SEARCH_K = int(os.getenv("SEARCH_K", "10"))

POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "postgres")

CONNECTION_STRING = (
    f"postgresql+psycopg://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)


def get_embeddings():
    """Retorna instância de embeddings configurada."""
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        raise ValueError("GOOGLE_API_KEY não definida")
    
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


def search_documents(query: str, k: int = None):
    """
    Realiza busca semântica (Requisito: k=10).
    
    Args:
        query: Pergunta do usuário
        k: Número de resultados (padrão: 10)
        
    Returns:
        Lista de (documento, score)
    """
    if k is None:
        k = SEARCH_K
    
    vectorstore = get_vectorstore()
    
    # similarity_search_with_score(query, k=10) conforme requisito
    return vectorstore.similarity_search_with_score(query, k=k)


def print_search_results(results):
    """Exibe resultados formatados."""
    if not results:
        print("❌ Nenhum resultado encontrado.")
        return
    
    print(f"📊 Encontrados {len(results)} resultados:")
    print()
    
    for i, (doc, score) in enumerate(results, 1):
        page = doc.metadata.get('page', '?')
        content = doc.page_content[:100].replace('\n', ' ')
        
        print(f"[{i}] Página {page} (Score: {score:.4f})")
        print(f"    {content}...")
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python src/search.py \"<sua pergunta>\"")
        sys.exit(1)
    
    query = " ".join(sys.argv[1:])
    
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║              BUSCA SEMÂNTICA                                  ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    print(f"🔎 Query: {query}")
    print()
    
    try:
        results = search_documents(query)
        print_search_results(results)
    except Exception as e:
        print(f"❌ ERRO: {e}")
        sys.exit(1)
