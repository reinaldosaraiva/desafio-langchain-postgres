#!/usr/bin/env python3
"""
Módulo de Busca Semântica

Funções importáveis:
- search_documents(): Realiza busca
- print_search_results(): Exibe resultados formatados
"""

import os
from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from typing import List, Tuple
from rich.console import Console
from rich.table import Table

console = Console()

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "campos_altos_edital_2025")
SEARCH_K = int(os.getenv("SEARCH_K", "10"))

POSTGRES_USER = os.getenv("POSTGRES_USER", "desafio_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "desafio_password")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "desafio_db")

CONNECTION_STRING = (
    f"postgresql+psycopg://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)


def get_vectorstore() -> PGVector:
    """Retorna instância do vectorstore configurada."""
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        raise ValueError("GOOGLE_API_KEY não definida")
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
        task_type="RETRIEVAL_QUERY",
    )
    
    return PGVector(
        embeddings=embeddings,
        connection=CONNECTION_STRING,
        collection_name=COLLECTION_NAME,
        use_jsonb=True,
    )


def search_documents(query: str, k: int = None) -> List[Tuple]:
    """
    Realiza busca semântica.
    
    Args:
        query: Pergunta do usuário
        k: Número de resultados
        
    Returns:
        Lista de (documento, score)
    """
    if k is None:
        k = SEARCH_K
    
    vectorstore = get_vectorstore()
    return vectorstore.similarity_search_with_score(query, k=k)


def print_search_results(results: List[Tuple], show_scores: bool = False):
    """
    Exibe resultados em tabela formatada.
    
    Args:
        results: Lista de (documento, score)
        show_scores: Se True, mostra coluna de scores
    """
    table = Table(title="Resultados da Busca")
    
    table.add_column("#", style="cyan", width=4)
    
    if show_scores:
        table.add_column("Score", style="magenta", width=10)
    
    table.add_column("Página", style="green", width=6)
    table.add_column("Conteúdo", style="white", width=60)
    
    for i, (doc, score) in enumerate(results, 1):
        page = doc.metadata.get('page', '?')
        content = doc.page_content[:100].replace('\n', ' ')
        
        if show_scores:
            table.add_row(str(i), f"{score:.4f}", str(page), f"{content}...")
        else:
            table.add_row(str(i), str(page), f"{content}...")
    
    console.print(table)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        console.print("[red]Uso: python src/search.py \"<query>\"[/red]")
        sys.exit(1)
    
    query = " ".join(sys.argv[1:])
    results = search_documents(query)
    print_search_results(results, show_scores=True)
