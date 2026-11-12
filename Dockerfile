FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    TORCH_HOME=/app/models/torch_cache \
    OPENCV_FFMPEG_CAPTURE_OPTIONS="timeout;3000000|rtsp_transport;tcp" \
    PORT=7860

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt \
    && python -c "import torchvision.models as m; m.mobilenet_v3_small(weights=m.MobileNet_V3_Small_Weights.DEFAULT)"

COPY core/ ./core/
COPY models/ ./models/
COPY web/ ./web/
COPY webapp/ ./webapp/
COPY web_app.py ./

RUN mkdir -p /app/alerts /app/reports /app/uploads /app/models/torch_cache \
    && chmod -R 777 /app/alerts /app/reports /app/uploads /app/models

EXPOSE 7860

CMD ["sh", "-c", "uvicorn web_app:app --host 0.0.0.0 --port ${PORT:-7860} --proxy-headers"]
