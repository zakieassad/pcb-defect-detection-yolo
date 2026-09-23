#!/bin/sh

uvicorn app.main:app \
  --host 127.0.0.1 \
  --port 8000 &

sleep 2

streamlit run frontend/app.py \
  --server.address 0.0.0.0 \
  --server.port ${PORT:-8080} \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --browser.gatherUsageStats false