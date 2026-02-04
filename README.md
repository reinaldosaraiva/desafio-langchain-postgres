# Desafio MBA Engenharia de Software com IA - Full Cycle
## Ingestão e Busca Semântica com LangChain, PostgreSQL e pgVector

Este projeto implementa um sistema RAG (Retrieval-Augmented Generation) para responder perguntas sobre documentos PDF utilizando busca semântica vetorial.

### Tecnologias

- **Python 3.9+**
- **LangChain** - Framework para aplicações LLM
- **LangChain Postgres** - Vector store com pgVector
- **Google Gemini** - Embeddings e LLM
- **PostgreSQL + pgVector** - Banco de dados vetorial

### Pré-requisitos

- Docker e Docker Compose
- Python 3.9 ou superior
- API Key do Google Gemini ([obter aqui](https://ai.google.dev/gemini-api/docs/api-key))

### Instalação

1. Clone do repositório
2. Copie o PDF para `document.pdf`
3. Configure o `.env`:
```bash
cp .env.example .env
# Edite .env e adicione sua GOOGLE_API_KEY
```

4. Suba o banco de dados:
```bash
docker compose up -d
```

5. Crie e ative o ambiente virtual:
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

6. Instale as dependências:
```bash
pip install -r requirements.txt
```

### Uso da CLI

Execute:

```bash
python main.py --help
```

#### Comandos Disponíveis

**1. Ingestão de PDF**
```bash
python main.py ingest document.pdf
```

**2. Busca Semântica**
```bash
python main.py search "Qual o valor estimado da contratação?"
```

**3. Chat Interativo**
```bash
python main.py chat
```

Digite 'sair' ou 'exit' para encerrar.

**4. Informações do Vectorstore**
```bash
python main.py info
```

### Exemplos de Uso

#### Exemplo 1: Ingestão
```bash
$ python main.py ingest document.pdf

╔═══════════════════════╗
║  INGESTÃO DE PDF        ║
╚═══════════════════════╝

Arquivo: document.pdf

Carregando PDF...
Dividindo documentos...
Inicializando vectorstore...
Ingestando chunks...

✓ 287 chunks ingeridos com sucesso!
```

#### Exemplo 2: Busca
```bash
$ python main.py search "Qual o valor estimado?"

╔═════════════════════╗
║  BUSCA SEMÂNTICA       ║
╚═════════════════════╝

Query: Qual o valor estimado?

┏━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ # ┃ Página ┃ Conteúdo                            ┃
┡━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 1 │ 1      │ R$175.414,67 (Cento e setenta e cinco... │
│ 2 │ 3      │ VALOR ESTIMADO DA CONTRATAÇÃO R$175...  │
└───┴────────┴────────┴──────────────────────────────────┘

Total: 10 resultados
```

#### Exemplo 3: Chat
```bash
$ python main.py chat

╔═════════════════════╗
║  CHAT INTERATIVO       ║
╚═════════════════════╝

Digite 'sair' ou 'exit' para encerrar.

Faça sua pergunta: Qual o valor estimado da contratação?

Pergunta: Qual o valor estimado da contratação?

┌──────────────────────────────────────────┐
│          RESPOSTA                      │
├──────────────────────────────────────────┤
│ O valor estimado da contratação é     │
│ R$175.414,67 (Cento e setenta e    │
│ cinco mil, quatrocentos e quatorze    │
│ reais e sessenta e sete centavos).    │
└──────────────────────────────────────────┘

Baseado em 10 contextos
```

### Casos de Teste

1. **Valor da contratação:** `python main.py search "Qual o valor estimado da contratação?"`
2. **Número do processo:** `python main.py search "Qual o número do processo licitatório?"`
3. **Fora do contexto:** `python main.py search "Qual é a capital da França?"`
4. **Opinião:** `python main.py search "Você acha esse edital bom ou ruim?"`

### Estrutura do Projeto

```
desafio-langchain-postgres/
├── docker-compose.yml      # PostgreSQL + pgVector
├── requirements.txt        # Dependências Python
├── .env.example          # Template de configuração
├── .gitignore
├── README.md             # Este arquivo
├── main.py              # CLI principal (sem Typer, compatível)
├── document.pdf          # PDF indexado (Edital Campos Altos)
└── src/
    ├── ingest.py         # Módulo de ingestão
    ├── search.py         # Módulo de busca
    └── chat.py          # Módulo de chat
```

### Configurações

| Variável | Padrão | Descrição |
|----------|---------|-----------|
| `CHUNK_SIZE` | 1000 | Tamanho do chunk em caracteres |
| `CHUNK_OVERLAP` | 150 | Sobreposição entre chunks |
| `SEARCH_K` | 10 | Número de resultados a buscar |
| `LLM_MODEL` | gemini-2.5-flash | Modelo de linguagem |
| `POSTGRES_PORT` | 5433 | Porta do PostgreSQL (evita conflito) |

### Licença

Este projeto é entregue como atividade acadêmica do MBA em Engenharia de Software com IA da Full Cycle.
