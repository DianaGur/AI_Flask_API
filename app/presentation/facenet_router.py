# match_router.py
from fastapi import APIRouter, HTTPException, Query
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
    embedding = embedding_store.get_embedding_by_user_id(req.userId)
    if embedding is None or embedding.size == 0:
        raise HTTPException(status_code=404, detail="User embedding not found.")

    results = matcher.match_user_to_candidates(
        userId=req.userId,
        candidate_ids=req.candidate_ids
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
    embedding = embedding_store.get_embedding_by_user_id(userId)
    if embedding is None or embedding.size == 0:
        raise HTTPException(status_code=404, detail="User embedding not found.")
    return {"userId": userId, "embedding": embedding.tolist()}

@router.get("/embeddings")
def get_all_embeddings(page: int = Query(1, ge=1), size: int = Query(10, ge=1, le=100)):
    embeddings, total = embedding_store.get_all_paginated(page=page, size=size)
    return {
        "total": total,
        "page": page,
        "size": size,
        "data": embeddings
    }

@router.delete("")
def delete_all_embeddings():
    result = embedding_store.delete_all()
    return {"deleted_count": result}