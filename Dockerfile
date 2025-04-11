FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directories
RUN mkdir -p models results logs tensorboard

# Copy code
COPY . .

# Environment variables
ENV PYTHONUNBUFFERED=1

# Use bash instead of direct python command for debugging
CMD ["/bin/bash"]
