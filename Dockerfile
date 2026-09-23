FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY frontend/requirements.txt ./frontend-requirements.txt

RUN pip install --no-cache-dir -r frontend-requirements.txt

COPY app ./app
COPY frontend ./frontend

COPY start.sh ./start.sh

RUN chmod +x ./start.sh

ENV MODEL_BUCKET=mma-cloudproject-tfi-grupo3
ENV MODEL_PREFIX=models/registry
ENV MODEL_DEVICE=cpu

CMD ["./start.sh"]