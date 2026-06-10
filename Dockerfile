# Dockerfile for "Memory Is a Hidden Dependency" Reproduction
# Stage 1: Base image with Python 3.10
# Lean compiler installation is commented out due to slow build times (>30 min).
# Enable it in a separate stage or manually if Lean verification is required.

FROM python:3.10-slim AS base

# Set environment variables to avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
# - git: for cloning repositories (if needed at runtime)
# - build-essential: for compiling Python C extensions
# - curl/wget: for downloading external tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set default workdir to /app/code as specified
WORKDIR /app/code

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
# Use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# --------------------------------------------------------
# Optional: Lean compiler installation (DISABLED by default)
# Uncomment the following lines to install Lean 4.
# Note: This step takes 30+ minutes and significantly increases image size.
# --------------------------------------------------------
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     curl \
#     git \
#     && rm -rf /var/lib/apt/lists/*
#
# RUN curl -L https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y \
#     && export PATH="$HOME/.elan/bin:$PATH" \
#     && elan default leanprover/lean4:v4.3.0 \
#     && rm -rf /root/.elan/toolchains/*/lib/lean/src/tests
# --------------------------------------------------------

# Copy all code into the container
# (Excluded files are handled by .dockerignore)
COPY . .

# Default command: print Python version (can be overridden with bash)
CMD ["python", "--version"]
