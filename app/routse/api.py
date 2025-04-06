from fastapi import APIRouter, UploadFile
from app.models.face_model import FaceEmbeddingModel
from app.services.clustering import cluster_embeddings

router = APIRouter()
model = FaceEmbeddingModel()

@router.post("/cluster")
async def cluster_faces(files: list[UploadFile]):
    embeddings = []
    for file in files:
        img = await file.read()
        embedding = model.get_embedding(img)
        embeddings.append(embedding)
    labels = cluster_embeddings(embeddings)
    return {"labels": labels}
