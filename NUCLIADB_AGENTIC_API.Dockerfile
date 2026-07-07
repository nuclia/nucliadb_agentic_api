FROM python:3.12 AS build

RUN pip install uv &&  apt update -y && apt install -y npm && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install dependencies
RUN uv venv /app
ENV VIRTUAL_ENV=/app
COPY . /app/.
RUN uv sync --active --frozen --directory /app --compile-bytecode

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