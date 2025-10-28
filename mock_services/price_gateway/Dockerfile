FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    pydantic==2.5.0 \
    requests==2.31.0

# Copy application files
COPY app.py /app/
COPY server.py /app/
COPY backend/ /app/backend/
COPY docker-entrypoint.sh /app/

# Make entrypoint executable
RUN chmod +x /app/docker-entrypoint.sh

# Expose gateway port (backend port 8090 is internal only)
EXPOSE 8082

# Health check (check gateway, which will check backend internally)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8082/health')"

# Run both backend and gateway
ENTRYPOINT ["/app/docker-entrypoint.sh"]
