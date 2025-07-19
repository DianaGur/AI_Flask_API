FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["gunicorn", "facenet_microserver_application:app", "--bind", "0.0.0.0:9001"]
# To run the Dockerfile in termina: 
    # docker build -t face-clustering-service .
    # docker run --env-file .env -p 8000:8000 face-clustering-service

