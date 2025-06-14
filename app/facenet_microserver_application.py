from fastapi import FastAPI
from presentation.facenet_router import router as facenet_router



app = FastAPI()
app.include_router(facenet_router, prefix="/facenet", tags=["facenet"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("facenet_microserver_application:app", host="0.0.0.0", port=9001, reload=True)
