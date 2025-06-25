# repository/embedding_store.py
from pymongo import MongoClient
from datetime import datetime
import numpy as np

class EmbeddingRepository:
    # יצירת החיבור למסד (כדאי להוציא בהמשך לקובץ config עם משתנים מהסביבה)
    client = MongoClient("mongodb://localhost:27017/")
    db = client["face_embeddings_db"]
    collection = db["user_embeddings"]


    def save_embedding(user_id: str, embedding: np.ndarray) -> str:
        """שומר וקטור embedding במסד"""
        embedding_list = embedding.tolist()  # המרה כדי שמונגו יקבל
        document = {
            "user_id": user_id,
            "embedding": embedding_list,
            "created_at": datetime.utcnow()
        }
        result = collection.insert_one(document)
        return str(result.inserted_id)


    def get_embedding_by_user_id(user_id: str) -> np.ndarray | None:
        """מאחזר embedding של משתמש לפי user_id"""
        doc = collection.find_one({"user_id": user_id})
        if doc:
            return np.array(doc["embedding"])
        return None


# Initialize the embedding store - global variable
embedding_store = EmbeddingRepository()
