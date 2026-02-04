#!/usr/bin/env python3
"""
CLI Principal usando Typer

Comandos disponíveis:
- ingest: Ingesta PDF no vectorstore
- search: Realiza busca semântica
- chat: Inicia chat interativo
- clear: Remove coleção do banco
- info: Exibe informações do vectorstore
"""

import os
import sys
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from ingest import ingest_pdf
from search import search_documents, print_search_results
from chat import chat_loop

console = Console()

app = typer.Typer(
    name="desafio-rag",
    help="CLI para Sistema RAG com LangChain + PostgreSQL + pgVector",
    add_completion=False,
)


@app.command()
def ingest(
    pdf_path: str = typer.Argument(
        "document.pdf",
        help="Caminho para o arquivo PDF",
        show_default=True,
    ),
    chunk_size: int = typer.Option(
        1000,
        "--chunk-size",
        "-c",
        help="Tamanho do chunk em caracteres",
        show_default=True,
    ),
    chunk_overlap: int = typer.Option(
        150,
        "--chunk-overlap",
        "-o",
        help="Sobreposição entre chunks em caracteres",
        show_default=True,
    ),
    collection_name: str = typer.Option(
        None,
        "--collection",
        "-n",
        help="Nome da coleção (default: usa .env)",
    ),
):
    """
    Ingesta PDF no vectorstore PostgreSQL.
    
    Divide o PDF em chunks, gera embeddings e armazena no banco.
    """
    try:
        if collection_name:
            os.environ["COLLECTION_NAME"] = collection_name
        
        console.print(Panel.fit(
            "[bold cyan]INGESTÃO DE PDF[/bold cyan]",
            border_style="cyan"
        ))
        console.print(f"\n[blue]Arquivo:[/blue] {pdf_path}")
        console.print(f"[blue]Chunk size:[/blue] {chunk_size}")
        console.print(f"[blue]Chunk overlap:[/blue] {chunk_overlap}")
        console.print(f"[blue]Coleção:[/blue] {collection_name or 'default'}\n")
        
        num_ingested = ingest_pdf(
            pdf_path=pdf_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        
        console.print(f"\n[bold green]✓ {num_ingested} chunks ingeridos com sucesso![/bold green]\n")
        
    except Exception as e:
        console.print(f"\n[bold red]✗ ERRO: {e}[/bold red]\n")
        raise typer.Exit(code=1)


@app.command()
def search(
    query: str = typer.Argument(
        ...,
        help="Pergunta ou consulta para buscar",
    ),
    k: int = typer.Option(
        10,
        "--k",
        "-n",
        help="Número de resultados a retornar",
        show_default=True,
    ),
    show_scores: bool = typer.Option(
        False,
        "--show-scores",
        "-s",
        help="Mostrar scores de similaridade",
    ),
):
    """
    Realiza busca semântica no vectorstore.
    
    Retorna os k documentos mais similares à query.
    """
    try:
        console.print(Panel.fit(
            f"[bold cyan]BUSCA SEMÂNTICA[/bold cyan]",
            border_style="cyan"
        ))
        console.print(f"\n[blue]Query:[/blue] {query}")
        console.print(f"[blue]Top k:[/blue] {k}\n")
        
        results = search_documents(query, k=k)
        print_search_results(results, show_scores=show_scores)
        
        console.print(f"\n[dim]Total: {len(results)} resultados[/dim]\n")
        
    except Exception as e:
        console.print(f"\n[bold red]✗ ERRO: {e}[/bold red]\n")
        raise typer.Exit(code=1)


@app.command()
def chat(
    max_history: int = typer.Option(
        10,
        "--max-history",
        "-m",
        help="Número máximo de mensagens no histórico",
        show_default=True,
    )
):
    """
    Inicia chat interativo com o documento.
    
    Faça perguntas e receba respostas baseadas no conteúdo do PDF.
    Digite 'sair' ou 'exit' para encerrar.
    """
    try:
        console.print(Panel.fit(
            "[bold cyan]CHAT INTERATIVO[/bold cyan]",
            border_style="cyan"
        ))
        console.print(f"\n[blue]Histórico máximo:[/blue] {max_history}")
        console.print("[yellow]Digite 'sair' ou 'exit' para encerrar.\n")
        
        chat_loop(max_history=max_history)
        
        console.print("\n[yellow]Chat encerrado.[/yellow]\n")
        
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Encerrando (Ctrl+C)...[/yellow]\n")
    except Exception as e:
        console.print(f"\n[bold red]✗ ERRO: {e}[/bold red]\n")
        raise typer.Exit(code=1)


@app.command()
def clear(
    collection_name: str = typer.Option(
        None,
        "--collection",
        "-n",
        help="Nome da coleção (default: usa .env)",
    ),
    confirm: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Confirmação automática (cuidado!)",
    )
):
    """
    Remove coleção do vectorstore.
    
    PERIGO: Esta operação é IRREVERSÍVEL!
    """
    try:
        if collection_name:
            os.environ["COLLECTION_NAME"] = collection_name
        
        if not confirm:
            console.print("\n[bold red]⚠  AVISO: Esta operação é IRREVERSÍVEL![/bold red]")
            console.print("[red]Todos os dados da coleção serão perdidos.[/red]\n")
            
            response = typer.confirm(
                "Tem certeza que deseja continuar?",
                default=False,
            )
            
            if not response:
                console.print("[yellow]Operação cancelada.[/yellow]\n")
                raise typer.Exit()
        
        from ingest import get_vectorstore, clear_collection
        vectorstore = get_vectorstore()
        clear_collection(vectorstore)
        
        console.print("[bold green]✓ Coleção removida com sucesso![/bold green]\n")
        
    except typer.Exit:
        pass
    except Exception as e:
        console.print(f"\n[bold red]✗ ERRO: {e}[/bold red]\n")
        raise typer.Exit(code=1)


@app.command()
def info():
    """
    Exibe informações do vectorstore.
    
    Mostra coleções, total de documentos, configurações.
    """
    try:
        from ingest import get_vectorstore
        
        console.print(Panel.fit(
            "[bold cyan]INFORMAÇÕES DO VECTORSTORE[/bold cyan]",
            border_style="cyan"
        ))
        
        vectorstore = get_vectorstore()
        
        table = Table(title="Configurações Atuais")
        table.add_column("Parâmetro", style="cyan")
        table.add_column("Valor", style="green")
        
        table.add_row("Coleção", os.getenv("COLLECTION_NAME", "default"))
        table.add_row("Embedding Model", "models/gemini-embedding-001")
        table.add_row("Vector Size", "3072")
        table.add_row("LLM Model", os.getenv("LLM_MODEL", "gemini-2.5-flash"))
        table.add_row("Search K", os.getenv("SEARCH_K", "10"))
        table.add_row("Chunk Size", os.getenv("CHUNK_SIZE", "1000"))
        table.add_row("Chunk Overlap", os.getenv("CHUNK_OVERLAP", "150"))
        
        console.print(table)
        console.print("\n[dim]Nota: Para obter estatísticas reais (nº de documentos, etc),")
        console.print("consulte diretamente o PostgreSQL.[/dim]\n")
        
    except Exception as e:
        console.print(f"\n[bold red]✗ ERRO: {e}[/bold red]\n")
        raise typer.Exit(code=1)


@app.command()
def version():
    """Exibe versão da CLI."""
    console.print("[bold cyan]desafio-rag[/bold cyan] v1.0.0")
    console.print("[dim]MBA Engenharia de Software com IA - Full Cycle[/dim]\n")


def main():
    """Ponto de entrada da CLI."""
    app()


if __name__ == "__main__":
    main()
