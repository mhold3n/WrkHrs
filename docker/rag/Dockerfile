FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    haystack-ai \
    qdrant-client \
    sentence-transformers \
    transformers \
    torch \
    pydantic \
    requests \
    numpy \
    rank-bm25 \
    nltk \
    scikit-learn

# Copy application code
COPY services/rag/ ./

# Create cache directory
RUN mkdir -p /cache

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:api", "--host", "0.0.0.0", "--port", "8000"]