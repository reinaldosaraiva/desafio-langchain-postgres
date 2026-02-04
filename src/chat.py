#!/usr/bin/env python3
"""
Módulo de Chat Interativo RAG

Funções importáveis:
- chat_loop(): Loop principal do chat
- ask_question(): Processa uma pergunta
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt

console = Console()

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "campos_altos_edital_2025")
SEARCH_K = int(os.getenv("SEARCH_K", "10"))
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "1.0"))

POSTGRES_USER = os.getenv("POSTGRES_USER", "desafio_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "desafio_password")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "desafio_db")

CONNECTION_STRING = (
    f"postgresql+psycopg://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

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
    """Retorna instância de LLM configurada."""
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        raise ValueError("GOOGLE_API_KEY não definida")
    
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=api_key,
        temperature=LLM_TEMPERATURE,
        max_retries=3,
        timeout=60,
    )


def get_vectorstore():
    """Retorna instância do vectorstore configurada."""
    api_key = os.getenv("GOOGLE_API_KEY")
    
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


def format_context(results):
    """Formata resultados para o prompt."""
    contexts = []
    
    for i, (doc, score) in enumerate(results, 1):
        content = doc.page_content
        context = f"[CONTEXTO {i}]\n{content}"
        contexts.append(context)
    
    return "\n\n".join(contexts)


def ask_question(pergunta: str):
    """
    Processa pergunta e retorna resposta.
    
    Args:
        pergunta: Pergunta do usuário
        
    Returns:
        Tupla (resposta, resultados)
    """
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search_with_score(pergunta, k=SEARCH_K)
    
    if not results:
        return "Não tenho informações necessárias para responder sua pergunta.", results
    
    contexto = format_context(results)
    prompt = RAG_PROMPT_TEMPLATE.format(contexto=contexto, pergunta=pergunta)
    
    llm = get_llm()
    resposta = llm.invoke(prompt)
    
    return resposta.content, results


def chat_loop(max_history: int = 10):
    """
    Loop principal do chat interativo.
    
    Args:
        max_history: Número máximo de mensagens no histórico
    """
    history = []
    
    while True:
        try:
            pergunta = Prompt.ask("\n[bold blue]Faça sua pergunta[/bold blue]")
            
            if pergunta.lower() in ['sair', 'exit', 'quit', '']:
                break
            
            console.print(f"\n[cyan]Pergunta:[/cyan] {pergunta}\n")
            
            resposta, results = ask_question(pergunta)
            
            resposta_panel = Panel(
                Markdown(resposta),
                title="[bold green]RESPOSTA[/bold green]",
                border_style="green",
                padding=(1, 2)
            )
            console.print(resposta_panel)
            
            console.print(f"\n[dim]Baseado em {len(results)} contextos[/dim]")
            
            history.append({"pergunta": pergunta, "resposta": resposta})
            if len(history) > max_history:
                history.pop(0)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"\n[red]ERRO: {e}[/red]")
            continue


if __name__ == "__main__":
    try:
        console.print("\n[bold cyan]╔═════════════════════════════════════════════════╗[/bold cyan]")
        console.print("[bold cyan]║  CLI INTERATIVO - BUSCA SEMÂNTICA RAG         ║[/bold cyan]")
        console.print("[bold cyan]╚═════════════════════════════════════════════════╝[/bold cyan]\n")
        console.print("[yellow]Digite 'sair' ou 'exit' para encerrar.[/yellow]\n")
        
        chat_loop()
        
        console.print("\n[yellow]Encerrando...[/yellow]\n")
        
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Encerrando (Ctrl+C)...[/yellow]\n")
