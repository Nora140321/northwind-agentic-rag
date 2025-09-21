from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def main():
    vs_path = "data/vectorstore"
    query = "What columns are in the 'Order Details' table and how is it related to Orders and Products?"

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(vs_path, embeddings, allow_dangerous_deserialization=True)

    docs = db.similarity_search(query, k=5)
    print(f"Top {len(docs)} results for: {query}\n")
    for i, d in enumerate(docs, 1):
        source = d.metadata.get("source", "")
        print(f"[{i}] {source}")
        print(d.page_content[:600].strip(), "\n")

if __name__ == "__main__":
    main()
