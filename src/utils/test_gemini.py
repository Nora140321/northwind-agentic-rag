import os
import traceback
from pathlib import Path

from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core import exceptions as gex


def main():
    # Resolve project root: src/utils/ -> src -> project
    DOTENV_PATH = Path(__file__).resolve().parents[2] / ".env"

    # Load .env with BOM-tolerant encoding to handle UTF-8 BOM issues on Windows
    loaded = load_dotenv(DOTENV_PATH, override=True, encoding="utf-8-sig")
    print(f"dotenv: {DOTENV_PATH} exists={DOTENV_PATH.exists()} loaded={loaded}")

    key = os.getenv("GOOGLE_API_KEY", "")
    mask = (key[:4] + "..." + key[-4:]) if key else ""
    print("KEY loaded:", bool(key), "len:", len(key), "mask:", mask)
    if not key:
        raise RuntimeError("GOOGLE_API_KEY is missing (failed to load from .env)")

    # Configure the Gemini client
    genai.configure(api_key=key)

    # Model to use for a quick text sanity check
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    print("Using model:", model_name)

    try:
        model = genai.GenerativeModel(model_name)
        r = model.generate_content("Reply with exactly: Gemini OK")
        text = (r.text or "").strip()
        print("Text response:", text)
        if text != "Gemini OK":
            raise RuntimeError(f"Unexpected text response: {text!r}")
    except gex.GoogleAPICallError as e:
        print("API error:", e.__class__.__name__, getattr(e, "code", ""), getattr(e, "message", str(e)))
        raise
    except Exception:
        traceback.print_exc()
        raise

    # Embeddings sanity check
    emb_model = os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")
    if not emb_model.startswith("models/"):
        emb_model = f"models/{emb_model}"
    print("Embedding model:", emb_model)

    try:
        emb = genai.embed_content(
            model=emb_model,
            content="Northwind sample database",
            task_type="retrieval_document",
        )
        # For text-embedding-004, embedding is a flat vector under "embedding"
        vector = emb.get("embedding") or emb
        if isinstance(vector, dict) and "embedding" in vector:
            vector = vector["embedding"]
        print("Embedding length:", len(vector))
    except gex.GoogleAPICallError as e:
        print("Embedding API error:", e.__class__.__name__, getattr(e, "code", ""), getattr(e, "message", str(e)))
        raise
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()