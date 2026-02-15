import subprocess
from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from fruit_classifier.data.datamodule import FruitDataModule
from fruit_classifier.training.lightning_module import FruitClassifierModule


class PlotSaver(Callback):
    def __init__(self):
        self.epochs = []
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.val_f1 = []

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        self.epochs.append(epoch)

        train_loss = trainer.callback_metrics.get("train/loss")
        if train_loss is not None:
            self.train_loss.append(float(train_loss))
        val_loss = trainer.callback_metrics.get("val/loss")
        if val_loss is not None:
            self.val_loss.append(float(val_loss))
        train_acc = trainer.callback_metrics.get("train/acc")
        if train_acc is not None:
            self.train_acc.append(float(train_acc))
        val_acc = trainer.callback_metrics.get("val/acc")
        if val_acc is not None:
            self.val_acc.append(float(val_acc))
        val_f1 = trainer.callback_metrics.get("val/f1")
        if val_f1 is not None:
            self.val_f1.append(float(val_f1))

    def on_train_end(self, trainer, pl_module):
        if not self.epochs:
            return

        plots_dir = Path("plots")
        plots_dir.mkdir(exist_ok=True)
        import matplotlib.pyplot as plt

        plt.figure()
        if self.train_loss:
            plt.plot(self.epochs, self.train_loss, label="train")
        if self.val_loss:
            plt.plot(self.epochs, self.val_loss, label="val")
        plt.legend()
        plt.title("Loss")
        plt.savefig(plots_dir / "loss.png")
        plt.close()

        plt.figure()
        if self.train_acc:
            plt.plot(self.epochs, self.train_acc, label="train")
        if self.val_acc:
            plt.plot(self.epochs, self.val_acc, label="val")
        plt.legend()
        plt.title("Accuracy")
        plt.savefig(plots_dir / "acc.png")
        plt.close()

        plt.figure()
        if self.val_f1:
            plt.plot(self.epochs, self.val_f1, label="val_f1")
        plt.legend()
        plt.title("F1 Score")
        plt.savefig(plots_dir / "f1.png")
        plt.close()

        print(f"Графики сохранены в {plots_dir}")


def ensure_data():
    data_path = Path("data/raw")
    if not data_path.exists() or not any(data_path.iterdir()):
        print("Данные не найдены. Загружаем...")
        from fruit_classifier.utils.download import download_data

        download_data()
    else:
        print("Данные уже есть.")


@hydra.main(
    version_base=None, config_path="fruit_classifier/configs", config_name="config"
)
def train(cfg: DictConfig):
    pl.seed_everything(cfg.seed, workers=True)

    ensure_data()

    dm = FruitDataModule(cfg)
    dm.setup()

    cfg.model.num_classes = dm.num_classes

    model = FruitClassifierModule(cfg)

    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.logging.experiment_name,
        tracking_uri=cfg.logging.mlflow_uri,
        run_name=cfg.logging.run_name,
    )
    mlflow_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
        mlflow_logger.log_hyperparams({"git_commit": commit})
    except Exception:
        pass

    checkpoint_callback = ModelCheckpoint(
        dirpath="models_artifacts/checkpoints",
        filename="fruit-{epoch:02d}-{val/acc:.4f}",
        monitor="val/acc",
        mode="max",
        save_top_k=2,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    plot_saver = PlotSaver()

    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator="auto",
        devices=1,
        logger=mlflow_logger,
        callbacks=[checkpoint_callback, lr_monitor, plot_saver],
        deterministic=True,
    )

    trainer.fit(model, dm)

    trainer.test(model, dm)


if __name__ == "__main__":
    train()
