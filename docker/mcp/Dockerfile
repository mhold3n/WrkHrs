FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    pydantic \
    requests \
    pyyaml \
    jsonschema \
    numpy \
    scipy \
    sympy \
    mendeleev \
    pubchempy \
    rdkit-pypi

# Copy application code
COPY services/mcp/ ./

# Create data directory
RUN mkdir -p /data

VOLUME ["/data"]

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:api", "--host", "0.0.0.0", "--port", "8000"]