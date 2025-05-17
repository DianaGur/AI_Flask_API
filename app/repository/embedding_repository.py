# repository/embedding_store.py
import numpy as np
from typing import Dict, List, Tuple

class EmbeddingRepository:
    def __init__(self):
        self._store: Dict[str, np.ndarray] = {}

    def save(self, user_id: str, embedding: np.ndarray):
        self._store[user_id] = embedding

    def get(self, user_id: str) -> np.ndarray:
        return self._store.get(user_id)

    def exists(self, user_id: str) -> bool:
        return user_id in self._store

    def get_many(self, user_ids: List[str]) -> Tuple[List[np.ndarray], List[str]]:
        embeddings = []
        ids = []
        for uid in user_ids:
            if uid in self._store:
                embeddings.append(self._store[uid])
                ids.append(uid)
        return embeddings, ids

# Initialize the embedding store - global variable
embedding_store = EmbeddingRepository()
