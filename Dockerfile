# Dockerfile
FROM python:3.12-slim

# Install uv (fast Python package manager) via the official distroless image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# System deps some Python packages (e.g. pdfplumber -> Pillow) may need at build time
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# Copy only dependency manifests first — Docker caches this layer until these files change
COPY pyproject.toml uv.lock ./

# Install dependencies matching the lockfile exactly (fails if lockfile is stale)
RUN uv sync --frozen --no-install-project --no-dev

# Now copy the actual application code (this layer rebuilds on every code change, deps don't)
COPY app ./app

# Install the project itself
RUN uv sync --frozen --no-dev

ENV PATH="/code/.venv/bin:$PATH"
ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.src.main:app --host 0.0.0.0 --port ${PORT:-8080}"]