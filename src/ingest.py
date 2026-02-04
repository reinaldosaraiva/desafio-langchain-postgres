#!/usr/bin/env python3
"""
Módulo de Ingestão de PDF para Vector Store

Funções importáveis:
- ingest_pdf(): Função principal de ingestão
- get_vectorstore(): Retorna instância do vectorstore
- clear_collection(): Remove coleção do banco
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "campos_altos_edital_2025")

POSTGRES_USER = os.getenv("POSTGRES_USER", "desafio_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "desafio_password")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "desafio_db")

CONNECTION_STRING = (
    f"postgresql+psycopg://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)


def load_pdf(file_path: str):
    """Carrega PDF e retorna documentos."""
    if not Path(file_path).exists():
        console.print(f"[red]ERRO: Arquivo não encontrado: {file_path}[/red]")
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
    
    loader = PyPDFLoader(file_path)
    return loader.load()


def split_documents(documents, chunk_size: int, chunk_overlap: int):
    """Divide documentos em chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )
    
    return text_splitter.split_documents(documents)


def get_embeddings():
    """Retorna instância de embeddings configurada."""
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        raise ValueError("GOOGLE_API_KEY não definida no .env")
    
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
        task_type="RETRIEVAL_DOCUMENT",
    )


def get_vectorstore():
    """Retorna instância do vectorstore configurado."""
    embeddings = get_embeddings()
    
    vectorstore = PGVector(
        embeddings=embeddings,
        connection=CONNECTION_STRING,
        collection_name=COLLECTION_NAME,
        use_jsonb=True,
    )
    
    return vectorstore


def clear_collection(vectorstore: PGVector):
    """Remove coleção do vectorstore."""
    # Deletar todos os documentos da coleção
    conn = vectorstore.conn
    cursor = conn.cursor()
    
    try:
        cursor.execute("DELETE FROM langchain_pg_embedding WHERE collection_id = %s", 
                     (vectorstore._collection_uuid,))
        conn.commit()
    finally:
        cursor.close()


def ingest_pdf(
    pdf_path: str = "document.pdf",
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> int:
    """
    Função principal de ingestão.
    
    Args:
        pdf_path: Caminho para o PDF
        chunk_size: Tamanho do chunk em caracteres
        chunk_overlap: Sobreposição entre chunks
        
    Returns:
        Número de chunks ingeridos
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Carregando PDF...", total=None)
        documents = load_pdf(pdf_path)
        progress.update(task, completed=True)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Dividindo documentos...", total=None)
        chunks = split_documents(documents, chunk_size, chunk_overlap)
        progress.update(task, completed=True)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Inicializando vectorstore...", total=None)
        vectorstore = get_vectorstore()
        progress.update(task, completed=True)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Ingestando chunks...", total=None)
        ids = vectorstore.add_documents(chunks)
        progress.update(task, completed=True)
    
    return len(ids)


if __name__ == "__main__":
    try:
        num_ingested = ingest_pdf()
        console.print(f"\n[bold green]✓ {num_ingested} chunks ingeridos![/bold green]\n")
    except Exception as e:
        console.print(f"\n[red]ERRO: {e}[/red]\n")
        sys.exit(1)
