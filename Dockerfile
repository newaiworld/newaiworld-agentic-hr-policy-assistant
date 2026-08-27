FROM python:3.11.15-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CHROMA_DIR=chroma_db
ENV CORPUS_DIR=corpus

COPY . .

RUN python -m pip install \
    --index-url https://download.pytorch.org/whl/cpu \
    "torch==2.13.0"

RUN ./build.sh

ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
