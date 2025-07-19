# services/image_service.py
import requests
from PIL import Image
from io import BytesIO
from typing import List
import numpy as np

from mtcnn import MTCNN  # module for face detection

def download_images(image_urls: List[str]) -> List[Image.Image]:
    images = []
    for url in image_urls:
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            images.append(image)
        except Exception as e:
            print(f"Error downloading the image from Firebase: {e}")
    return images

def detect_faces(image: Image.Image) -> List[Image.Image]:
    detector = MTCNN()
    image_np = np.array(image)
    detections = detector.detect_faces(image_np)
    faces = []

    for detection in detections:
        x, y, w, h = detection['box']
        face_crop = image.crop((x, y, x + w, y + h))
        faces.append(face_crop)

    return faces
