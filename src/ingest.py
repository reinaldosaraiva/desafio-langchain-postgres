#!/usr/bin/env python3
"""
Script de ingestão do PDF para Vector Store PostgreSQL + pgVector

Uso:
  python src/ingest.py
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

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "default")

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
    """Retorna instância de embeddings configurada (modelo obrigatório)."""
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        raise ValueError("GOOGLE_API_KEY não definida no .env")
    
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",  # Modelo obrigatório
        google_api_key=api_key,
        task_type="RETRIEVAL_DOCUMENT",
    )


def ingest_pdf(
    pdf_path: str = "document.pdf",
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
):
    """
    Função principal de ingestão.
    
    Requisitos:
    - Chunks de 1000 caracteres
    - Overlap de 150 caracteres
    - Cada chunk convertido em embedding
    - Vetores armazenados no PostgreSQL + pgVector
    """
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║           INGESTÃO DO PDF PARA VECTOR STORE                   ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    
    # 1. Carregar PDF
    print(f"📄 Carregando PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"✓ PDF carregado: {len(documents)} páginas")
    print()
    
    # 2. Dividir em chunks (Requisito: 1000 chars, 150 overlap)
    print("✂️  Dividindo documentos em chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✓ Chunks gerados: {len(chunks)} (size: {chunk_size}, overlap: {chunk_overlap})")
    print()
    
    # 3. Inicializar vectorstore
    print("💾 Inicializando Vector Store (PostgreSQL + pgVector)...")
    embeddings = get_embeddings()
    
    vectorstore = PGVector(
        embeddings=embeddings,
        connection=CONNECTION_STRING,
        collection_name=COLLECTION_NAME,
        use_jsonb=True,
    )
    print("✓ Vector Store inicializado")
    print()
    
    # 4. Ingestar chunks (converte cada chunk em embedding)
    print("📊 Ingestando chunks e gerando embeddings...")
    ids = vectorstore.add_documents(chunks)
    print(f"✓ {len(ids)} chunks ingeridos com embeddings gerados")
    print()
    
    return len(ids)


if __name__ == "__main__":
    try:
        num_ingested = ingest_pdf()
        print("╔══════════════════════════════════════════════════════════════╗")
        print(f"║           INGESTÃO CONCLUÍDA: {num_ingested} CHUNKS             ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        print()
    except Exception as e:
        print(f"\n❌ ERRO: {e}\n")
        sys.exit(1)
