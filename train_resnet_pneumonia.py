import argparse
import copy
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


@dataclass
class EpochResult:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_transforms(img_size: int = 224) -> Dict[str, transforms.Compose]:
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(size=img_size, scale=(0.85, 1.0)),
            transforms.RandomRotation(degrees=7),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    eval_tf = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )
    return {"train": train_tf, "val": eval_tf, "test": eval_tf}


def build_dataloaders(
    dataset_root: Path,
    batch_size: int,
    num_workers: int,
    img_size: int,
) -> Tuple[Dict[str, DataLoader], Dict[str, int], Dict[int, str]]:
    tf = build_transforms(img_size=img_size)

    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"
    test_dir = dataset_root / "test"

    for d in [train_dir, val_dir, test_dir]:
        if not d.exists():
            raise FileNotFoundError(f"資料夾不存在: {d}")

    image_datasets = {
        "train": datasets.ImageFolder(root=str(train_dir), transform=tf["train"]),
        "val": datasets.ImageFolder(root=str(val_dir), transform=tf["val"]),
        "test": datasets.ImageFolder(root=str(test_dir), transform=tf["test"]),
    }

    # 確保三個 split 的標籤順序一致
    train_class_to_idx = image_datasets["train"].class_to_idx
    for split in ["val", "test"]:
        if image_datasets[split].class_to_idx != train_class_to_idx:
            raise ValueError(
                f"{split} 類別索引與 train 不一致: {image_datasets[split].class_to_idx} != {train_class_to_idx}"
            )

    dataloaders = {
        split: DataLoader(
            image_datasets[split],
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        for split in ["train", "val", "test"]
    }

    idx_to_class = {v: k for k, v in train_class_to_idx.items()}
    return dataloaders, train_class_to_idx, idx_to_class


def compute_class_weights(train_loader: DataLoader, num_classes: int, device: torch.device) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for _, labels in train_loader: # (images, labels) = batch
        for i in range(num_classes):
            counts[i] += (labels == i).sum() # counts = [<num_of_class0>, <num_of_class1>, ...]
    counts = torch.clamp(counts, min=1.0) 
    weights = counts.sum() / (num_classes * counts)
    return weights.to(device)


def build_model(model_name: str, num_classes: int, freeze_backbone: bool) -> nn.Module:
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"不支援的模型: {model_name}")

    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False
    return model


def binary_metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor, positive_index: int = 1) -> Tuple[int, int, int, int]:
    preds = logits.argmax(dim=1)

    tp = ((preds == positive_index) & (labels == positive_index)).sum().item()
    tn = ((preds != positive_index) & (labels != positive_index)).sum().item()
    fp = ((preds == positive_index) & (labels != positive_index)).sum().item()
    fn = ((preds != positive_index) & (labels == positive_index)).sum().item()
    return tp, tn, fp, fn


def summarize_metrics(loss_sum: float, samples: int, tp: int, tn: int, fp: int, fn: int) -> EpochResult:
    loss = loss_sum / max(samples, 1)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-8)
    return EpochResult(loss=loss, accuracy=accuracy, precision=precision, recall=recall, f1=f1)


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer = None,
    positive_index: int = 1,
) -> EpochResult:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_samples = 0
    total_tp = total_tn = total_fp = total_fn = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        tp, tn, fp, fn = binary_metrics_from_logits(logits.detach(), labels, positive_index=positive_index)
        total_tp += tp
        total_tn += tn
        total_fp += fp
        total_fn += fn

    return summarize_metrics(total_loss, total_samples, total_tp, total_tn, total_fp, total_fn)


def train(
    dataset_root: Path,
    output_dir: Path,
    model_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    num_workers: int,
    img_size: int,
    seed: int,
    freeze_backbone: bool,
    unfreeze_epoch: int,
) -> None:
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")

    dataloaders, class_to_idx, idx_to_class = build_dataloaders(
        dataset_root=dataset_root,
        batch_size=batch_size,
        num_workers=num_workers,
        img_size=img_size,
    )
    num_classes = len(class_to_idx)
    if num_classes != 2:
        raise ValueError(f"此任務預期 2 類，但偵測到 {num_classes} 類: {class_to_idx}")

    positive_index = class_to_idx.get("PNEUMONIA", 1)
    print(f"類別映射: {class_to_idx}, 正類別索引(PNEUMONIA): {positive_index}")

    model = build_model(model_name=model_name, num_classes=num_classes, freeze_backbone=freeze_backbone).to(device)
    class_weights = compute_class_weights(dataloaders["train"], num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_state = None
    best_val_f1 = -1.0
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        if freeze_backbone and epoch == unfreeze_epoch:
            print("解凍 backbone，進行全網路微調...")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = AdamW(model.parameters(), lr=lr * 0.1, weight_decay=weight_decay)
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

        train_metrics = run_one_epoch(
            model=model,
            loader=dataloaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            positive_index=positive_index,
        )
        val_metrics = run_one_epoch(
            model=model,
            loader=dataloaders["val"],
            criterion=criterion,
            optimizer=None,
            device=device,
            positive_index=positive_index,
        )

        scheduler.step(val_metrics.loss)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics.loss,
                "train_acc": train_metrics.accuracy,
                "train_f1": train_metrics.f1,
                "val_loss": val_metrics.loss,
                "val_acc": val_metrics.accuracy,
                "val_f1": val_metrics.f1,
                "val_precision": val_metrics.precision,
                "val_recall": val_metrics.recall,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        print(
            f"[Epoch {epoch:02d}/{epochs}] "
            f"Train loss={train_metrics.loss:.4f}, acc={train_metrics.accuracy:.4f}, f1={train_metrics.f1:.4f} | "
            f"Val loss={val_metrics.loss:.4f}, acc={val_metrics.accuracy:.4f}, f1={val_metrics.f1:.4f}"
        )

        if val_metrics.f1 > best_val_f1:
            best_val_f1 = val_metrics.f1
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, output_dir / "best_model.pth")

    if best_state is None:
        raise RuntimeError("訓練未產生可用模型。")

    model.load_state_dict(best_state)
    test_metrics = run_one_epoch(
        model=model,
        loader=dataloaders["test"],
        criterion=criterion,
        optimizer=None,
        device=device,
        positive_index=positive_index,
    )

    print(
        "\n[Test] "
        f"loss={test_metrics.loss:.4f}, acc={test_metrics.accuracy:.4f}, "
        f"precision={test_metrics.precision:.4f}, recall={test_metrics.recall:.4f}, f1={test_metrics.f1:.4f}"
    )

    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    metadata = {
        "model_name": model_name,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "img_size": img_size,
        "best_val_f1": best_val_f1,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\n模型與紀錄已輸出到: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chest X-ray Pneumonia Classification with ResNet Transfer Learning")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=r"C:\Users\Edward\Desktop\Classifier\chest_xray",
        help="資料集根目錄，底下應包含 train/ val/ test",
    )
    parser.add_argument("--output_dir", type=str, default=r"C:\Users\Edward\Desktop\Classifier\outputs")
    parser.add_argument("--model_name", type=str, choices=["resnet18", "resnet50"], default="resnet18")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0, help="Windows 建議先用 0，穩定後可調高")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze_backbone", action="store_true", help="前幾個 epoch 先只訓練最後分類層")
    parser.add_argument("--unfreeze_epoch", type=int, default=4, help="若啟用 freeze_backbone，於此 epoch 解凍 backbone")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        dataset_root=Path(args.dataset_root),
        output_dir=Path(args.output_dir),
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        img_size=args.img_size,
        seed=args.seed,
        freeze_backbone=args.freeze_backbone,
        unfreeze_epoch=args.unfreeze_epoch,
    )

