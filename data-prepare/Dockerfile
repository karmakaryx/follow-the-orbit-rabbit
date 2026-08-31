FROM python:3.11-slim

WORKDIR /app

RUN apt update && apt install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ingestion.py .
COPY preprocessing.py .

CMD ["python", "ingestion.py"]
