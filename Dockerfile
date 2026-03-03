FROM python:3.11-slim

# System deps for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Pre-train baseline model during build
RUN python pretrain.py --samples 200 --epochs 30

# Expose port
ENV PORT=10000
EXPOSE 10000

# Run
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
