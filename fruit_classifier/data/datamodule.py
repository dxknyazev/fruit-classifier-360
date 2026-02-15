import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from fruit_classifier.data.dataset import FruitDataset
from fruit_classifier.utils.paths import get_data_path


class FruitDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.data_dir = get_data_path()
        self.batch_size = cfg.train.batch_size
        self.num_workers = cfg.train.num_workers
        self.train_transform = self._get_train_transform()
        self.val_test_transform = self._get_val_test_transform()
        self.num_classes = None

    def _get_train_transform(self):
        aug = self.cfg.data.augmentation
        return transforms.Compose(
            [
                transforms.Resize((self.cfg.data.img_size, self.cfg.data.img_size)),
                transforms.RandomRotation(aug.rotation),
                transforms.ColorJitter(
                    brightness=aug.brightness, contrast=aug.contrast
                ),
                transforms.RandomAffine(
                    degrees=0, translate=(aug.translate, aug.translate)
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _get_val_test_transform(self):
        return transforms.Compose(
            [
                transforms.Resize((self.cfg.data.img_size, self.cfg.data.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def setup(self, stage=None):
        full_dataset = FruitDataset(
            self.data_dir / "Training", transform=self.train_transform
        )
        test_dataset = FruitDataset(
            self.data_dir / "Test", transform=self.val_test_transform
        )

        self.num_classes = len(full_dataset.classes)

        val_len = int(self.cfg.data.val_split * len(full_dataset))
        train_len = len(full_dataset) - val_len
        self.train_dataset, self.val_dataset = random_split(
            full_dataset,
            [train_len, val_len],
            generator=torch.Generator().manual_seed(self.cfg.seed),
        )
        self.test_dataset = test_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
