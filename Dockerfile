# Use official Python 3.10 slim image as base
FROM python:3.10-slim

# Install system dependencies (git, curl, unzip) and AWS CLI dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl unzip \
    && rm -rf /var/lib/apt/lists/*

# Install AWS CLI v2
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf awscliv2.zip aws/

# Set working directory
WORKDIR /app

# Copy requirements.txt and install Python dependencies
COPY api/requirements.txt ./
RUN pip install --upgrade pip \
    && pip install "numpy<2"

# Install PyTorch CPU version (2.1.2+cpu)
RUN pip install --no-cache-dir torch==2.1.2+cpu torchvision==0.16.2+cpu torchaudio==2.1.2+cpu \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Install other dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy API source code (excluding saved_model)
COPY api/ ./

# Copy entrypoint script and make it executable
COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh

# Environment variables for S3 model download
ENV AWS_DEFAULT_REGION=us-west-1 \
    S3_BUCKET=cinebert-mlflow-models \
    S3_MODEL_PATH=saved_model

# Entrypoint to download model and start the API
ENTRYPOINT ["./entrypoint.sh"]

# Default command to run FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
