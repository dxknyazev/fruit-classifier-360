import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from fruit_classifier.training.lightning_module import (  # noqa: E402
    FruitClassifierModule,
)

original_load = torch.load


def patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return original_load(*args, **kwargs)


torch.load = patched_load


@hydra.main(
    version_base=None, config_path="../fruit_classifier/configs", config_name="config"
)
def export_onnx(cfg: DictConfig):
    checkpoint_dir = Path("models_artifacts/checkpoints")
    checkpoints = list(checkpoint_dir.glob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError("No checkpoint found. Train first.")
    checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)

    model = FruitClassifierModule.load_from_checkpoint(checkpoint_path)
    model.eval()

    model.cpu()

    try:
        img_size = model.hparams.cfg.data.img_size
    except AttributeError:
        img_size = 100
        print(f"Warning: could not get img_size from hparams, using default {img_size}")

    dummy_input = torch.randn(1, 3, img_size, img_size).cpu()

    onnx_path = checkpoint_dir.parent / "model.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"ONNX model saved to {onnx_path}")


if __name__ == "__main__":
    export_onnx()
