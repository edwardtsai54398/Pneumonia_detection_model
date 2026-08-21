from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from constant import IMAGENET_MEAN, IMAGENET_STD


def download_dataset():
    """Download the Kaggle chest-xray-pneumonia dataset via kagglehub.

    Returns the Path to the extracted `chest_xray/` directory
    (containing train/val/test subfolders).
    """
    import kagglehub

    dataset_root = Path(kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia"))
    dataset_root = dataset_root / "chest_xray"

    for split in ["train", "val", "test"]:
        split_dir = dataset_root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"資料夾不存在: {split_dir}")

    return dataset_root


def build_transforms(img_size=224):
    train_tf = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(size=img_size, scale=(0.85, 1.0)),
            transforms.RandomRotation(degrees=7),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    eval_tf = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    return {"train": train_tf, "val": eval_tf, "test": eval_tf}


def build_datasets(dataset_root, img_size=224):
    """Build train/val/test ImageFolder datasets with consistent class_to_idx."""
    dataset_root = Path(dataset_root)
    tf = build_transforms(img_size=img_size)

    image_datasets = {}
    for split in ["train", "val", "test"]:
        split_dir = dataset_root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"資料夾不存在: {split_dir}")
        image_datasets[split] = ImageFolder(root=str(split_dir), transform=tf[split])

    train_class_to_idx = image_datasets["train"].class_to_idx
    for split in ["val", "test"]:
        if image_datasets[split].class_to_idx != train_class_to_idx:
            raise ValueError(
                f"{split} 類別索引與 train 不一致: {image_datasets[split].class_to_idx} != {train_class_to_idx}"
            )

    idx_to_class = {v: k for k, v in train_class_to_idx.items()}
    return image_datasets, train_class_to_idx, idx_to_class


def make_dataloaders(datasets, batch_size, num_workers=2):
    return {
        split: DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        for split in datasets
    }
