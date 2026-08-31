FROM python:3.11-slim

WORKDIR /app

RUN apt update && apt install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY sequence_builder.py .
COPY torch_dataset.py .
COPY model.py .
COPY train.py .
COPY serve.py .

# 학습은 docker-compose에서 command를 "python train.py"로 오버라이드해서 사용
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
