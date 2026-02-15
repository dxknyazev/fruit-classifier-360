#!/bin/bash
# Запуск MLflow сервера для MLflow-модели
poetry run mlflow models serve -m models_artifacts/mlflow_model --env-manager=local -p 5000
