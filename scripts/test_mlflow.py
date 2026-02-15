import argparse
from pathlib import Path

import numpy as np
import requests
import torchvision.transforms as transforms
from PIL import Image


def get_class_names(data_path="data/raw/Training"):
    """Возвращает отсортированный список имён классов из папки Training"""
    train_dir = Path(data_path)
    if not train_dir.exists():
        train_dir = Path("data/raw/Training")
    classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    return classes


def preprocess(image_path):
    """Препроцессинг изображения под размер 100x100 и нормализацию"""
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    tensor = transform(img).unsqueeze(0).numpy()
    return tensor.tolist()


def softmax(logits):
    """Вычисляет вероятности из логитов (для численной устойчивости)"""
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / exp_logits.sum()


def test_mlflow(image_path, url="http://127.0.0.1:5000/invocations"):
    class_names = get_class_names()
    print(f"Found {len(class_names)} classes. First few: {class_names[:5]}")

    input_tensor = preprocess(image_path)
    payload = {"inputs": input_tensor}
    response = requests.post(url, json=payload)

    if response.status_code == 200:
        result = response.json()
        logits = result["predictions"]["output"][0]
        logits = np.array(logits)

        pred_class_idx = int(np.argmax(logits))
        pred_class_name = (
            class_names[pred_class_idx]
            if pred_class_idx < len(class_names)
            else "Unknown"
        )
        print(f"\nPredicted class: {pred_class_name} (index {pred_class_idx})")

        probs = softmax(logits)
        print("\nTop-5 probabilities:")
        top5 = np.argsort(probs)[-5:][::-1]
        for i, idx in enumerate(top5):
            class_name = class_names[idx] if idx < len(class_names) else f"Class_{idx}"
            print(f"  {i+1}. {class_name}: {probs[idx]:.4f}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to image")
    args = parser.parse_args()
    test_mlflow(args.image)
