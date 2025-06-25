# match_router.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from services.face_similarity_service import FaceSimilearityService
from services.image_service import download_images, detect_faces
from repository.embedding_repository import embedding_store

router = APIRouter()

# מבנה הבקשה להתאמה
class MatchRequest(BaseModel):
    userId: str
    candidate_ids: List[str]

# Boundary for saveing the request
class SaveEmbeddingRequest(BaseModel):
    userId: str
    image_urls: List[str]

# Main service of the server
matcher = FaceSimilearityService()

@router.post("/match")
def match_faces(req: MatchRequest):
    if not embedding_store.exists(req.userId):
        raise HTTPException(status_code=404, detail="User embedding not found.")

    results = matcher.match_user_to_candidates(
        userId=req.userId,
        candidate_ids=req.candidateIds
    )
    if not results:
        raise HTTPException(status_code=400, detail="No candidate embeddings found.")

    return {"matches": results}

@router.post("/embeddings")
def save_user_embedding(req: SaveEmbeddingRequest):
    images = download_images(req.image_urls)
    if not images:
        raise HTTPException(status_code=400, detail="No valid images downloaded.")

    faces = []
    for img in images:
        faces += detect_faces(img)
    if not faces:
        raise HTTPException(status_code=400, detail="No faces detected.")

    matcher.save_embedding(req.userId, faces)
    return {"message": f"Embedding for user {req.userId} saved."}

@router.get("/embeddings/{userId}")
def get_user_embedding(userId: str):
    if not embedding_store.exists(userId):
        raise HTTPException(status_code=404, detail="User embedding not found.")
    emb = embedding_store.get(userId).tolist()
    return {"userId": userId, "embedding": emb}
