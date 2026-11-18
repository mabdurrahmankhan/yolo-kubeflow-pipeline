FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libxcb1 && \
    rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir ultralytics onnx onnxruntime pillow numpy
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"