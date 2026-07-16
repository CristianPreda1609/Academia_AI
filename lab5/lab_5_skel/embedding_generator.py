import json
import os
from document_chunker import load_n_chunk_docs
from embeddings_client import EmbeddingsClient
def embedding_generator():
    if os.path.exists("embeddings.json"):
        print("Embeddings already exist. Skipping generation.")
        return

    embedding_list = []
    chunks = load_n_chunk_docs()
    embeddings_client = EmbeddingsClient()
    for chunk in chunks:
        embedded = embeddings_client.get_embedding(chunk["content"])
        chunk_with_embedding = {
            "document_id": chunk["document_id"],
            "chunk_index": chunk["chunk_index"],
            "content": chunk["content"],
            "embedding": embedded
        }
        embedding_list.append(chunk_with_embedding)

    with open(os.path.join(".", "embeddings.json"), "w", encoding="utf-8") as f:
        json.dump(embedding_list, f, ensure_ascii=False, indent=4)
