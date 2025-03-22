from fastapi import FastAPI
frm app.routes import api

app = FastAPI()
app.include_router(api.router)