FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir .
COPY tts_service/ ./tts_service/
EXPOSE 8004
CMD ["python", "-m", "tts_service"]
