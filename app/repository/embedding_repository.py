# repository/embedding_store.py
from pymongo import MongoClient
from datetime import datetime
import numpy as np

class EmbeddingRepository:
    # יצירת החיבור למסד (כדאי להוציא בהמשך לקובץ config עם משתנים מהסביבה)
    client = MongoClient("mongodb://localhost:27017/")
    db = client["face_embeddings_db"]
    collection = db["user_embeddings"]


    def save(self, userId: str, embedding: np.ndarray) -> str:
        # save user embedding to MongoDB
        embedding_list = embedding.tolist()  # המרה כדי שמונגו יקבל
        document = {
            "userId": userId,
            "embedding": embedding_list,
            "created_at": datetime.utcnow()
        }
        result = self.collection.insert_one(document)
        return str(result.inserted_id)


    def get_embedding_by_user_id(self, userId: str) -> np.ndarray | None:
        #returns user embedding by userId
        doc = self.collection.find_one({"userId": userId})
        if doc:
            return np.array(doc["embedding"])
        return None
    
    def get_many_embeddings_by_user_ids(self, user_ids: list[str]) -> tuple[list[np.ndarray], list[str]]:
        docs = self.collection.find({"userId": {"$in": user_ids}})
        
        embeddings = []
        id_mapping = []

        for doc in docs:
            embeddings.append(np.array(doc["embedding"]))
            id_mapping.append(doc["userId"])

        return embeddings, id_mapping



# Initialize the embedding store - global variable
embedding_store = EmbeddingRepository()
