#!/bin/bash
set -e

echo "Downloading model from S3..."
mkdir -p /app/saved_model
aws s3 cp s3://cinebert-mlflow-models/saved_model/ /app/saved_model/ --recursive

echo "Starting FastAPI server..."
exec uvicorn main:app --host 0.0.0.0 --port 8000
