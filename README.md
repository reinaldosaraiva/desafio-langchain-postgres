# Ingestão e Busca Semântica com LangChain e PostgreSQL

## Objetivo

Você deve entregar um software capaz de:

**Ingestão:** Ler um arquivo PDF e salvar suas informações em um banco de dados PostgreSQL com extensão pgVector.

**Busca:** Permitir que o usuário faça perguntas via linha de comando (CLI) e receba respostas baseadas apenas no conteúdo do PDF.

## Tecnologias Obrigatórias

- **Linguagem:** Python
- **Framework:** LangChain
- **Banco de dados:** PostgreSQL + pgVector
- **Execução do banco de dados:** Docker & Docker Compose

## Instalação

1. Clone do repositório
2. Configure o `.env`:
```bash
cp .env.example .env
# Edite .env e adicione sua GOOGLE_API_KEY
```

3. Suba o banco de dados:
```bash
docker compose up -d
```

4. Crie e ative o ambiente virtual:
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

5. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Ordem de Execução

### 1. Subir o banco de dados

```bash
docker compose up -d
```

### 2. Executar ingestão do PDF

```bash
python src/ingest.py
```

### 3. Rodar o chat

```bash
python src/chat.py
```

## Exemplo no CLI

### Pergunta no Contexto

```
Faça sua pergunta:

PERGUNTA: Qual o valor estimado da contratação?
RESPOSTA: O valor estimado da contratação é R$175.414,67 (Cento e setenta e cinco mil, quatrocentos e quatorze reais e sessenta e sete centavos).
```

### Pergunta Fora do Contexto

```
Faça sua pergunta:

PERGUNTA: Qual é a capital da França?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

## Pacotes Utilizados

```python
# Split
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings (Gemini)
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# PDF
from langchain_community.document_loaders import PyPDFLoader

# Ingestão
from langchain_postgres import PGVector

# Busca
similarity_search_with_score(query, k=10)
```

## Requisitos Atendidos

### 1. Ingestão do PDF

✅ **PDF dividido em chunks de 1000 caracteres com overlap de 150.**
✅ **Cada chunk convertido em embedding (modelos/embedding-001).**
✅ **Vetores armazenados no banco de dados PostgreSQL com pgVector.**

### 2. Consulta via CLI

✅ **Script Python para simular um chat no terminal.**

**Passos ao receber uma pergunta:**
1. ✅ Vetorizar a pergunta.
2. ✅ Buscar os 10 resultados mais relevantes (k=10) no banco vetorial.
3. ✅ Montar o prompt e chamar a LLM (gemini-2.5-flash-lite).
4. ✅ Retornar a resposta ao usuário.

## Estrutura do Projeto

```
desafio-langchain-postgres/
├── docker-compose.yml      # PostgreSQL + pgVector
├── requirements.txt        # Dependências
├── .env.example          # Template da variável GOOGLE_API_KEY
├── src/
│   ├── ingest.py         # Script de ingestão do PDF
│   ├── search.py         # Script de busca
│   ├── chat.py           # CLI para interação com usuário
├── document.pdf          # PDF para ingestão
└── README.md             # Instruções de execução
```

## Prompt Utilizado

```plaintext
CONTEXTO:
{resultados concatenados do banco de dados}

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
{pergunta do usuário}

RESPONDA A "PERGUNTA DO USUÁRIO"
```

## Licença

Este projeto é entregue como atividade acadêmica do MBA em Engenharia de Software com IA da Full Cycle.
