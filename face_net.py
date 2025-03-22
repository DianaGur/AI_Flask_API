class FaceEmbeddingModel:
    def __init__(self):
        self.model = load_model()  # יכול להיות keras או torch

    def get_embedding(self, face_image):
        # החזרת embedding לווקטור
        return self.model.predict(face_image)
