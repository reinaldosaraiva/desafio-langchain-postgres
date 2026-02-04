#!/usr/bin/env python3
"""
Script de Testes do Desafio RAG
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ingest import ingest_pdf
from search import search_documents
from chat import ask_question

from rich.console import Console
from rich.panel import Panel

console = Console()


def test_case(num, question, expected_in_answer=True):
    """Executa um caso de teste"""
    console.print(f"\n[bold yellow]Caso de Teste {num}[/bold yellow]")
    console.print(f"[cyan]Pergunta:[/cyan] {question}")
    console.print(f"[dim]Esperado:[/dim] Resposta válida" if expected_in_answer else "[dim]Esperado:[/dim] Não tenho informações")
    
    try:
        resposta, results = ask_question(question)
        
        resposta_panel = Panel(
            resposta[:300],
            title="[bold green]RESPOSTA[/bold green]",
            border_style="green",
        )
        console.print(resposta_panel)
        
        # Verificar se respondeu "Não tenho informações"
        tem_info = "não tenho informações necessárias" not in resposta.lower()
        
        if expected_in_answer:
            status = "✅ PASS" if tem_info else "❌ FAIL (Recusou indevidamente)"
        else:
            status = "✅ PASS" if not tem_info else "❌ FAIL (Deveria recusar)"
            
        console.print(f"[bold]{status}[/bold]")
        
    except Exception as e:
        console.print(f"[bold red]❌ ERRO: {e}[/bold red]")


def main():
    """Executa todos os testes"""
    console.print("[bold cyan]╔══════════════════════════════════════════════════════════════╗[/bold cyan]")
    console.print("[bold cyan]║  TESTES DO SISTEMA RAG - DESAFIO MBA FULL CYCLE        ║[/bold cyan]")
    console.print("[bold cyan]╚══════════════════════════════════════════════════════════════╝[/bold cyan]")
    
    test_cases = [
        # Caso 1: Pergunta dentro do contexto (Valor)
        (1, "Qual o valor estimado da contratação?", True),
        
        # Caso 2: Pergunta dentro do contexto (Processo)
        (2, "Qual o número do processo licitatório?", True),
        
        # Caso 3: Pergunta fora do contexto (Capital França)
        (3, "Qual é a capital da França?", False),
        
        # Caso 4: Pergunta de opinião (Deve recusar)
        (4, "Você acha esse edital bom ou ruim?", False),
    ]
    
    for num, question, expected in test_cases:
        test_case(num, question, expected)
        console.print("[dim]" + "-"*60 + "[/dim]")
    
    console.print("\n[bold green]TESTES CONCLUÍDOS[/bold green]\n")


if __name__ == "__main__":
    main()
