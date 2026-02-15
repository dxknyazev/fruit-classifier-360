import argparse

import numpy as np
import torchvision.transforms as transforms
import tritonclient.http as httpclient
from PIL import Image


def preprocess(image_path):
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    tensor = transform(img).unsqueeze(0).numpy()
    return tensor


def infer(image_path, url="localhost:8000", model_name="fruit_classifier"):
    client = httpclient.InferenceServerClient(url=url)
    inputs = [httpclient.InferInput("input", [1, 3, 100, 100], "FP32")]
    inputs[0].set_data_from_numpy(preprocess(image_path))
    outputs = [httpclient.InferRequestedOutput("output")]
    response = client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
    probs = response.as_numpy("output")[0]
    return probs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to image")
    args = parser.parse_args()
    probs = infer(args.image)
    print("Probabilities:", probs)
    print("Predicted class:", np.argmax(probs))
