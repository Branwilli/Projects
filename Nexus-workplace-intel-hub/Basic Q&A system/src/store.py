import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve(query_embedding, doc_embeddings, chunks, top_k=3):
    score = [cosine_similarity(query_embedding, emb) for emb in doc_embeddings]
    top_indices = np.argsort(score)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]