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
