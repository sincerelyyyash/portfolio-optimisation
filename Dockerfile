FROM python:3.9-slim

WORKDIR /app

# Install system dependencies first (good practice)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p models results logs tensorboard

# Copy application code
COPY . .

# Environment variables
ENV PYTHONUNBUFFERED=1

# Command to run
CMD ["python", "portfolio_optimization.py"]
