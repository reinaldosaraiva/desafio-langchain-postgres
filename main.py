#!/usr/bin/env python3
"""
Script Principal do Desafio RAG

Uso:
  python main.py ingest --help
  python main.py ingest document.pdf
  python main.py search "Qual o valor?"
  python main.py chat
"""

import sys
import os

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ingest import ingest_pdf, get_vectorstore, clear_collection
from search import search_documents, print_search_results
from chat import chat_loop

from rich.console import Console

console = Console()


def cmd_ingest(args):
    """Comando ingest"""
    try:
        from rich.panel import Panel
        
        console.print(Panel.fit(
            "[bold cyan]INGESTÃO DE PDF[/bold cyan]",
            border_style="cyan"
        ))
        
        pdf_path = args[0] if args else "document.pdf"
        
        console.print(f"\n[blue]Arquivo:[/blue] {pdf_path}")
        
        num_ingested = ingest_pdf(
            pdf_path=pdf_path,
            chunk_size=1000,
            chunk_overlap=150,
        )
        
        console.print(f"\n[bold green]✓ {num_ingested} chunks ingeridos com sucesso![/bold green]\n")
    except Exception as e:
        console.print(f"\n[red]ERRO: {e}[/red]\n")
        sys.exit(1)


def cmd_search(args):
    """Comando search"""
    try:
        from rich.panel import Panel
        
        if not args:
            console.print("[red]Uso: python main.py search \"<query>\"[/red]")
            sys.exit(1)
        
        query = args[0]
        
        console.print(Panel.fit(
            f"[bold cyan]BUSCA SEMÂNTICA[/bold cyan]",
            border_style="cyan"
        ))
        console.print(f"\n[blue]Query:[/blue] {query}")
        
        results = search_documents(query, k=10)
        print_search_results(results, show_scores=True)
        
        console.print(f"\n[dim]Total: {len(results)} resultados[/dim]\n")
    except Exception as e:
        console.print(f"\n[red]ERRO: {e}[/red]\n")
        sys.exit(1)


def cmd_chat(args):
    """Comando chat"""
    try:
        from rich.panel import Panel
        
        console.print(Panel.fit(
            "[bold cyan]CHAT INTERATIVO[/bold cyan]",
            border_style="cyan"
        ))
        console.print("\n[yellow]Digite 'sair' ou 'exit' para encerrar.[/yellow]\n")
        
        chat_loop(max_history=10)
        
        console.print("\n[yellow]Chat encerrado.[/yellow]\n")
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Encerrando...[/yellow]\n")
    except Exception as e:
        console.print(f"\n[red]ERRO: {e}[/red]\n")
        sys.exit(1)


def cmd_info(args):
    """Comando info"""
    try:
        from rich.table import Table
        from rich.panel import Panel
        
        console.print(Panel.fit(
            "[bold cyan]INFORMAÇÕES DO VECTORSTORE[/bold cyan]",
            border_style="cyan"
        ))
        
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
        console.print("\n[dim]Nota: Para obter estatísticas reais,")
        console.print("consulte diretamente o PostgreSQL.[/dim]\n")
    except Exception as e:
        console.print(f"\n[red]ERRO: {e}[/red]\n")
        sys.exit(1)


def main():
    """Função principal"""
    if len(sys.argv) < 2:
        console.print("[bold cyan]Desafio MBA Engenharia de Software com IA - Full Cycle[/bold cyan]")
        console.print("\n[bold]Uso:[/bold]")
        console.print("  python main.py ingest [arquivo.pdf]")
        console.print("  python main.py search \"<query>\"")
        console.print("  python main.py chat")
        console.print("  python main.py info")
        sys.exit(1)
    
    command = sys.argv[1]
    args = sys.argv[2:]
    
    commands = {
        'ingest': cmd_ingest,
        'search': cmd_search,
        'chat': cmd_chat,
        'info': cmd_info,
    }
    
    if command not in commands:
        console.print(f"[red]Comando desconhecido: {command}[/red]")
        console.print(f"[dim]Comandos disponíveis: {', '.join(commands.keys())}[/dim]")
        sys.exit(1)
    
    commands[command](args)


if __name__ == "__main__":
    main()
