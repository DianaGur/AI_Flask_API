# services/faiss_service.py
import numpy as np
import faiss
from repository.embedding_repository import get_embedding_by_user_id, get_many_embeddings_by_user_ids

class FaissService:
    def __init__(self):
        self.dim = 128 

    def match_user_to_candidates(self, user_id: str, candidate_ids: list[str], top_k: int = 10) -> list[dict]:
        # get users embedding
        query_embedding = get_embedding_by_user_id(user_id)
        if query_embedding is None:
            return []

        query_embedding = query_embedding.astype('float32').reshape(1, -1)

        # Get all candidates embeddings
        candidate_embeddings, id_mapping = get_many_embeddings_by_user_ids(candidate_ids)
        if not candidate_embeddings:
            return []

        # preparing to faiss
        candidate_embeddings = np.array(candidate_embeddings).astype('float32')
        index = faiss.IndexFlatIP(self.dim)
        index.add(candidate_embeddings)

        # serch for similar embeddings
        similarities, indices = index.search(query_embedding, k=min(top_k, len(candidate_embeddings)))

        # bulding result json
        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                "user_id": id_mapping[idx],
                "similarity": float(similarities[0][i])
            })

        return results
