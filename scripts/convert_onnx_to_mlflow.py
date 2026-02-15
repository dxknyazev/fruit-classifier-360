from pathlib import Path

import mlflow
import mlflow.onnx
import onnx

onnx_path = Path("models_artifacts/model.onnx")
if not onnx_path.exists():
    raise FileNotFoundError(f"ONNX model not found at {onnx_path}")

onnx_model = onnx.load(str(onnx_path))

mlflow_model_path = Path("models_artifacts/mlflow_model")
mlflow.onnx.save_model(onnx_model, str(mlflow_model_path))

print(f"✅ MLflow model saved to {mlflow_model_path}")
print(
    "Now you can run: mlflow models serve -m models_artifacts/mlflow_model --env-manager=local -p 5000"
)
