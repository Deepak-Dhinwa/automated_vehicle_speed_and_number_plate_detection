# Use official Python base image with GPU support (if deploying with GPU)
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Alternatively, if GPU not needed in container, use python base:
# FROM python:3.10

# Set working directory
WORKDIR /app

# Install basic dependencies
RUN apt-get update && \
    apt-get install -y git curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python
RUN apt-get update && apt-get install -y python3-pip

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire app directory
COPY ./app ./app
COPY ./sort ./sort

# Expose port (FastAPI default)
EXPOSE 8000

# Set environment variables if needed (optional)
ENV PYTHONUNBUFFERED=1

# Command to run the FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
