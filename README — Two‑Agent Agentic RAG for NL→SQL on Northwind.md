
# Two‑Agent Agentic RAG for NL→SQL on Northwind

## Overview
This project provides a two‑agent system that converts natural‑language business questions into SQL and executes them against the Northwind SQLite database. It uses Retrieval‑Augmented Generation (RAG) with FAISS to ground the LLM in schema notes and business rules, minimizing hallucinations.

- Agent 2 (Planner): Generates SQL from user prompts using Gemini, grounded by FAISS‑retrieved context
- Agent 1 (Executor): Executes SQL on SQLite and returns formatted results

Extras:
- One‑shot mode and interactive REPL
- CSV/Markdown exports and quick charts
- Conversation memory for multi‑turn refinement (ConversationBufferMemory)

## Tech Stack
- Python 3.x
- LangChain (agents, memory, vectorstores)
- Google Gemini (chat + embeddings)
- SQLite (Northwind)
- FAISS for vector search
- Rich, Matplotlib, Pandas for UX/exports

## Project Structure
```
northwind-agentic-rag/
├─ data/
│  ├─ 1751978414.sqlite3            # Northwind DB
│  └─ .env.ini                      # API key and config
├─ docs/
│  └─ schema/
│     └─ schema.md                  # Table notes, business rules, examples
├─ out/                              # Exports (CSV/MD/PNG)
├─ two_agent_langchain_app.py        # Main app (two agents + RAG)
└─ .gitignore
```

## Setup

1) Python environment
```
pip install langchain langchain-community langchain-google-genai google-generativeai faiss-cpu sqlalchemy python-dotenv pandas matplotlib rich
```

2) Configure environment in `data/.env.ini`:
```
GOOGLE_API_KEY=AIzaSyBtm-eNo8v0AE1FTZuZyacVAxEDOGTkmZI
GEMINI_MODEL=gemini-1.5-flash
GEMINI_EMBEDDING_MODEL=models/text-embedding-004
DATABASE_PATH=C:\Users\Noura\OneDrive\Desktop\northwind-agentic-rag\data\1751978414.sqlite3
VECTOR_DIR=data\vectorstore_lc
SCHEMA_DIR=docs\schema
```

3) Add schema docs in `docs/schema/schema.md` (see template in the main guide). The vector index builds automatically on first run or when `data/vectorstore_lc` is deleted.

## How to Run

One‑shot:
```
python two_agent_langchain_app.py --q "Monthly revenue totals for 2016. Columns Month, TotalRevenue. Order by Month. Do not include LIMIT." --rows 50 --csv out\monthly_revenue_2016.csv --md out\monthly_revenue_2016.md --plot out\monthly_revenue_2016.png --strip-limit
```

REPL:
```
python two_agent_langchain_app.py --repl --rows 50 --strip-limit
# type your questions
# commands: exit | quit | clear
```

## Required Analytics — Examples

- Top 5 countries by total sales
```
"Top 5 countries by total revenue. Include LIMIT 5. Columns Country, TotalRevenue. Order by TotalRevenue desc."
```

- Best‑selling products by quantity
```
"Best-selling products by quantity overall. Columns ProductName, TotalQuantity. Order by TotalQuantity desc. Include LIMIT 10."
```

- Revenue by category and quarter (2016)
```
"Revenue by category by quarter for 2016. Columns Quarter, CategoryName, TotalRevenue. Order by Quarter, TotalRevenue desc."
```

- Average order value per customer
```
"Average order value per customer overall. Columns CustomerID, CompanyName, AvgOrderValue. Order by AvgOrderValue desc. Include LIMIT 10."
```

- Sales trends over years
```
"Total revenue by year. Columns Year, TotalRevenue. Order by Year."
```

Export artifacts with `--csv`, `--md`, and `--plot`.

## How RAG Improves SQL Accuracy
- The model retrieves schema notes, canonical joins, and business rules (e.g., revenue formula) from FAISS and uses them in prompts.
- This grounding significantly reduces wrong table/column names and ensures consistent date handling via `STRFTIME` on `datetime(Orders.OrderDate)`.

## Notes and Tips
- To rebuild embeddings after changing `docs/schema/`, delete `data/vectorstore_lc`.
- If you need strict “top N”, avoid `--strip-limit` in CLI or explicitly request `LIMIT N`.
- Security: keep your `.env.ini` out of version control and rotate keys after demos.

## Demo Script (≤ 2 minutes)
1. Show the repo structure and `data/.env.ini` (mask the key).
2. Display `docs/schema/schema.md` highlights (joins, revenue formula).
3. Run a one‑shot query with CSV/MD/plot export.
4. Start REPL, ask a follow‑up (multi‑turn), then `clear` and `exit`.
5. Briefly explain RAG grounding and why it prevents hallucination.

## License
Educational use.
