# services/face_match_service.py
import numpy as np
import os
import faiss
from embedding_repository import embedding_store
from facenet import FaceNetBackbone
import torch
import torch.nn.functional as F

class FaceSimilearityService:
    def __init__(self):
        self.model = FaceNetBackbone(embedding_size=128)
        self.model.load_state_dict(torch.load("app/models/facenet_model.pth", map_location=torch.device('cpu')))
        self.model.eval()

    def save_embedding(self, userId: str, faces: list):
        embeddings = []
        for face in faces:
            face = face.resize((160, 160))
            face_tensor = torch.tensor(np.array(face)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            with torch.no_grad():
                emb = self.model(face_tensor).squeeze().numpy()
                embeddings.append(emb)
        final_embedding = np.mean(embeddings, axis=0).astype('float32')
        embedding_store.save(userId, final_embedding)
        print(f"Saved embedding for user {userId} with shape {final_embedding.shape}")

    def match_user_to_candidates(self, userId: str, candidate_ids: list):
        query_embedding = embedding_store.get_embedding_by_user_id(userId).reshape(1, -1)
        candidate_embeddings, id_mapping = embedding_store.get_many_embeddings_by_user_ids(candidate_ids)

        if not candidate_embeddings:
            return []

        candidate_embeddings = np.array(candidate_embeddings).astype('float32')
        dim = query_embedding.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(candidate_embeddings)
        similarities, indices = index.search(query_embedding, k=min(10, len(candidate_embeddings)))

        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                "userId": id_mapping[idx],
                "similarity": float(similarities[0][i])
            })

        return results
