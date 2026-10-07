# API server. Mount or download the fine-tuned model at /app/results, or set
# PHONEME_MODEL_ID to a Hugging Face Hub repo; otherwise the public ASR fallback is used.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY download_nltk.py .
RUN python download_nltk.py

COPY api.py db.py word_list.py sample_history.json ./
COPY pronunciation ./pronunciation
COPY src ./src
COPY Docs/model ./Docs/model

ENV PRACTICE_DB_PATH=/data/practice_history.db
VOLUME ["/data"]
EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
