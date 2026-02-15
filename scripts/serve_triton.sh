#!/bin/bash
# Подготовка модели для Triton и запуск сервера
mkdir -p models/fruit_classifier/1
cp models_artifacts/model.onnx models/fruit_classifier/1/
cp scripts/triton_config.pbtxt models/fruit_classifier/config.pbtxt

# Запуск Triton Server в Docker
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/models:/models \
  nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver --model-repository=/models
