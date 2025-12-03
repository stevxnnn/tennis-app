FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY tennis_rotation.py .
COPY assets/ ./assets/

EXPOSE 8080

CMD ["python", "tennis_rotation.py"]

