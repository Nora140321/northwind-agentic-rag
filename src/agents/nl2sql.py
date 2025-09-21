# src/agents/nl2sql.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


def _load_env() -> None:
    dotenv_path = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(dotenv_path, override=True, encoding="utf-8-sig")


def _build_embeddings() -> GoogleGenerativeAIEmbeddings:
    model_name = os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")
    if not model_name.startswith("models/"):
        model_name = f"models/{model_name}"
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY is missing in environment")
    return GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=key)


def _load_retriever():
    project_root = Path(__file__).resolve().parents[2]
    index_dir = project_root / "data" / "index" / "faiss"
    emb = _build_embeddings()
    vectordb = FAISS.load_local(str(index_dir), embeddings=emb, allow_dangerous_deserialization=True)
    return vectordb.as_retriever(search_kwargs={"k": 4})


def _build_llm() -> ChatGoogleGenerativeAI:
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY is missing in environment")
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=key,
        temperature=0.1,
        max_output_tokens=1024,
    )
    return llm


PROMPT_SYS = """You are an expert SQLite SQL generator for the Northwind database.
You MUST output a single SQLite SELECT query only, with no extra commentary, no code fences, and NO trailing semicolon.

Rules:
- Use only tables and columns that exist in the provided CONTEXT.
- Quote identifiers that contain spaces with double quotes, e.g., "Order Details".
- Revenue formula: sum("Order Details".UnitPrice * "Order Details".Quantity * (1 - "Order Details".Discount))
- Countries use Orders.ShipCountry. Dates use Orders.OrderDate.
- Prefer INNER JOIN when required. Common joins:
  Orders.OrderID = "Order Details".OrderID
  "Order Details".ProductID = Products.ProductID
  Products.CategoryID = Categories.CategoryID
  Orders.CustomerID = Customers.CustomerID
  Orders.EmployeeID = Employees.EmployeeID
  Orders.ShipVia = Shippers.ShipperID
- SQLite date helpers:
  Year: CAST(strftime('%Y', Orders.OrderDate) AS INTEGER)
  Month: CAST(strftime('%m', Orders.OrderDate) AS INTEGER)
  Quarter: CAST(((CAST(strftime('%m', Orders.OrderDate) AS INTEGER) + 2) / 3) AS INTEGER)
- When averaging order values per customer, FIRST compute per-order totals in a subquery or CTE, THEN average per customer.
  Never use nested aggregates like AVG(SUM(...)) in a single SELECT.
- Always include ORDER BY when returning top-N and apply LIMIT accordingly if the user asks for top results.
- If the user asks for a count or aggregate, ensure appropriate GROUP BY.
- Do not use backticks or T-SQL syntax.
- Do NOT include any explanation, only the SQL query.
"""

PROMPT_USER_TEMPLATE = """CONTEXT:
{context}

QUESTION:
{question}

Return only the SQL query (no trailing semicolon).
"""


def _sanitize_sql(text: str) -> str:
    s = (text or "").strip()
    # strip code fences if any
    if s.startswith("```"):  
        s = s.strip("`")  
        if s.lower().startswith("sql"):  
            s = s[3:].lstrip()  
    if s.endswith("```"):
        s = s[:-3].rstrip()
    # remove trailing semicolon
    while s.endswith(";"):
        s = s[:-1].rstrip()
    return s


@dataclass
class NL2SQL:
    retriever: any
    llm: ChatGoogleGenerativeAI

    @classmethod
    def from_env(cls) -> "NL2SQL":
        _load_env()
        retriever = _load_retriever()
        llm = _build_llm()
        return cls(retriever=retriever, llm=llm)

    def _retrieve_context(self, question: str) -> str:
        # Use invoke() to avoid deprecation warnings
        docs = self.retriever.invoke(question)
        return "\n\n---\n\n".join(d.page_content for d in docs)

    def generate_sql(self, question: str) -> str:
        context = self._retrieve_context(question)
        messages = [
            SystemMessage(content=PROMPT_SYS),
            HumanMessage(content=PROMPT_USER_TEMPLATE.format(context=context, question=question)),
        ]
        resp = self.llm.invoke(messages)
        return _sanitize_sql(getattr(resp, "content", "") or "")

    def revise_sql(self, question: str, prev_sql: str, error_message: str) -> str:
        """Ask the model to correct the previous SQL given the SQLite error."""
        context = self._retrieve_context(question)

        hints = ""
        if "misuse of aggregate function" in error_message.lower():
            hints = """
HINTS:
- Do not nest aggregates like AVG(SUM(...)) in one SELECT.
- Compute inner sums (e.g., per-order totals) in a subquery or CTE and then aggregate (AVG/COUNT/SUM) in the outer query.
Example pattern:
WITH order_totals AS (
  SELECT Orders.CustomerID AS CustomerID,
         Orders.OrderID AS OrderID,
         SUM("Order Details".UnitPrice * "Order Details".Quantity * (1 - "Order Details".Discount)) AS order_total
  FROM Orders
  INNER JOIN "Order Details" ON Orders.OrderID = "Order Details".OrderID
  GROUP BY Orders.OrderID
)
SELECT CustomerID, AVG(order_total) AS AverageOrderValue
FROM order_totals
GROUP BY CustomerID
"""

        sys = PROMPT_SYS
        user = f"""CONTEXT:
{context}

QUESTION:
{question}

The previous SQL caused an error and needs correction:

SQL:
{prev_sql}

ERROR:
{error_message}
{hints}

Return only the corrected SQL query (no trailing semicolon).
"""
        resp = self.llm.invoke([SystemMessage(content=sys), HumanMessage(content=user)])
        return _sanitize_sql(getattr(resp, "content", "") or "")