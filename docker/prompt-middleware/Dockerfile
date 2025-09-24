FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    pydantic \
    numpy \
    torch \
    transformers \
    sentence-transformers \
    librosa \
    soundfile \
    networkx \
    scipy \
    matplotlib \
    asyncio \
    concurrent.futures

# Copy application code
COPY services/prompt_middleware/ ./app/

# Create logs directory
RUN mkdir -p /logs

# Expose port
EXPOSE 8002

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONIOENCODING=utf-8

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# Run the application
CMD ["python", "app/app.py"]
