FROM python:3.12 AS build

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    pip install uv && apt update -y && apt install -y npm

# Install dependencies first (cached unless manifests change)
RUN uv venv /app
ENV VIRTUAL_ENV=/app
COPY pyproject.toml uv.lock /app/
COPY agents/nucliadb/pyproject.toml /app/agents/nucliadb/pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --active --frozen --directory /app --compile-bytecode --no-install-workspace --link-mode=copy

# Copy source code and reinstall workspace packages
COPY . /app/.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --active --frozen --directory /app --compile-bytecode --link-mode=copy

#
# Only copy the virtual env to the final image.
#
FROM python:3.12
COPY --from=build /app /app
ENV PATH=/app/bin:$PATH

# OCI metadata: links this image to the GitHub repo as a package, with source + license.
LABEL org.opencontainers.image.source="https://github.com/nuclia/nucliadb_agentic_api" \
    org.opencontainers.image.description="Agentic on top of NucliaDB" \
    org.opencontainers.image.licenses="Apache-2.0" \
    org.opencontainers.image.authors="Nuclia Team"