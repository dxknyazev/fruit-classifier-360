#!/bin/bash
# Конвертация ONNX в TensorRT (требуется trtexec)
ONNX_PATH="models_artifacts/model.onnx"
ENGINE_PATH="models_artifacts/model.engine"

if [ ! -f "$ONNX_PATH" ]; then
    echo "ONNX file not found. Run export_onnx.py first."
    exit 1
fi

trtexec --onnx=$ONNX_PATH --saveEngine=$ENGINE_PATH --fp16 \
        --minShapes=input:1x3x100x100 \
        --optShapes=input:8x3x100x100 \
        --maxShapes=input:16x3x100x100 \
        --workspace=2048
