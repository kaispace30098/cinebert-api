# 🎬 CineBERT API

Sentiment classification API using DistilBERT, deployed via Docker on AWS EC2.

---

## 🚀 Quick Start: Deploy to EC2

### 🔐 Step 0: SSH into your EC2

```bash
ssh -i "your-key.pem" ubuntu@<your-ec2-public-ip>
```

---

### 📥 Step 1: Navigate & Pull the Latest Code

```bash
cd cinebert-api
git pull
```

---

### 🛠 Step 2: Build and Run the Docker Container

```bash
# Stop & remove previous container if needed
sudo docker stop cinebert-api || true
sudo docker rm cinebert-api || true

# Build the new image
sudo docker build -t cinebert-api:latest .

# Run the container (automatically restart if EC2 reboots)
sudo docker run -d --name cinebert-api \
  --restart unless-stopped \
  -p 8000:8000 \
  cinebert-api:latest
```

---

### ✅ Step 3: Test Your API

Visit the following URL in your browser:

```
http://<your-ec2-public-ip>:8000/docs
```

You’ll see the **FastAPI Swagger UI**, where you can test the `/predict` and `/health` endpoints.

---

## 📦 Project Structure

```
cinebert-api/
├── api/
│   ├── main.py              # FastAPI app
│   ├── requirements.txt     # API dependencies
│   └── saved_model/         # (downloaded from S3)
├── Dockerfile               # For building the image
├── entrypoint.sh            # Downloads model from s3 & starts the server 
└── .github/workflows/       # (Optional CI setup)
```

---

## 📁 Model Download (via entrypoint.sh)

The Docker container automatically downloads your fine-tuned model from this S3 path:

```
s3://cinebert-mlflow-models/saved_model/
```

Make sure:

- The EC2 instance role or environment has S3 read permission.
- The path is correct and contains HuggingFace-style saved model files.

---

## 🧯 Common Commands

```bash
# Check container logs
sudo docker logs -f cinebert-api

# Restart container manually
sudo docker restart cinebert-api

# View running containers
sudo docker ps
```

---

## 🔒 Security Note

Do **not** commit the following to GitHub or write them here:

- `.pem` private key
- `.env` secrets (API keys, AWS credentials)
- Actual model files (they are pulled from S3)