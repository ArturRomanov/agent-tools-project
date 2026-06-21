FROM python:3.12-slim
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir .
COPY stt_service/ ./stt_service/
EXPOSE 8003
CMD ["python", "-m", "stt_service"]
