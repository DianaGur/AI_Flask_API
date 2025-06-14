import subprocess
import sys

required_packages = [
    "fastapi", "uvicorn", "pydantic", "torch", "torchvision",
    "facenet-pytorch", "faiss-cpu", "numpy", "typing-extensions",
    "scikit-learn", "matplotlib", "Pillow", "mtcnn", "tensorflow"
]

for package in required_packages:
    try:
        __import__(package.replace("-", ""))
    except ImportError:
        print(f"[INFO] Installing missing package: {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

from fastapi import FastAPI
from presentation.facenet_router import router as facenet_router



app = FastAPI()
app.include_router(facenet_router, prefix="/facenet", tags=["facenet"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("facenet_microserver_application:app", host="0.0.0.0", port=9001, reload=True)
