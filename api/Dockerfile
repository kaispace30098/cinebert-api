FROM python:3.10

RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install "numpy<2"

# install torch 2.1+ explicitly
RUN pip install --no-cache-dir torch==2.1.2+cpu torchvision==0.16.2+cpu torchaudio==2.1.2+cpu \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]



# FROM python:3.10

# # 安裝系統依賴與 AWS CLI
# RUN apt-get update && apt-get install -y \
#     git curl unzip \
#     && rm -rf /var/lib/apt/lists/*

# # 安裝 AWS CLI v2 (官方安裝方式)
# RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
#     unzip awscliv2.zip && \
#     ./aws/install && \
#     rm -rf awscliv2.zip aws/

# WORKDIR /app

# # 複製 requirements.txt 並安裝 python package
# COPY api/requirements.txt ./

# RUN pip install --upgrade pip \
#     && pip install "numpy<2"

# # 安裝 PyTorch cpu 版本（依你版本）
# RUN pip install --no-cache-dir torch==2.1.2+cpu torchvision==0.16.2+cpu torchaudio==2.1.2+cpu \
#     -f https://download.pytorch.org/whl/cpu/torch_stable.html

# RUN pip install --no-cache-dir -r requirements.txt

# # 複製所有程式碼（不包含 saved_model）
# COPY api/ .

# # 複製 entrypoint 腳本
# COPY entrypoint.sh .

# # 讓 entrypoint.sh 可執行
# RUN chmod +x entrypoint.sh

# # 環境變數（可用 docker run -e 代替）
# ENV AWS_DEFAULT_REGION=us-east-1 \
#     S3_BUCKET=your-s3-bucket-name \
#     S3_MODEL_PATH=cine-bert/saved_model

# # 啟動時先從 S3 下載模型，再啟動 uvicorn
# ENTRYPOINT ["./entrypoint.sh"]
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
