import argparse
import json
import logging
import os
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

logger = logging.getLogger(__name__)


def _setup_logging(output_dir: Path) -> None:
    """同時輸出到 console 與 log 檔案，含時間戳與等級。"""
    fmt = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    console = logging.StreamHandler()
    console.setFormatter(fmt)

    file_handler = logging.FileHandler(output_dir / "training.log", encoding="utf-8")
    file_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(console)
    root.addHandler(file_handler)


# ── dataclass ────────────────────────────────────────────────────────────────

@dataclass
class EpochResult:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0


# ── 種子與可重現性 ──────────────────────────────────────────────────────────

def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # [改進 #17] deterministic 模式改為可選，預設關閉以保留 GPU 效能
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


# ── 資料增強 ─────────────────────────────────────────────────────────────────

def build_transforms(img_size: int = 224) -> Dict[str, transforms.Compose]:
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    # [改進 #9] 降低水平翻轉機率；加入亮度/對比度等更適合醫學影像的增強
    train_tf = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(size=img_size, scale=(0.85, 1.0)),
            transforms.RandomRotation(degrees=7),
            transforms.RandomHorizontalFlip(p=0.15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
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


# ── 資料載入 ─────────────────────────────────────────────────────────────────

def _build_datasets(
    dataset_root: Path,
    img_size: int,
) -> Tuple[Dict[str, datasets.ImageFolder], Dict[str, int], Dict[int, str]]:
    """建立 train / val 資料集（test 延遲載入）。"""
    tf = build_transforms(img_size=img_size)

    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"

    for d in [train_dir, val_dir]:
        if not d.exists():
            raise FileNotFoundError(f"資料夾不存在: {d}")

    image_datasets: Dict[str, datasets.ImageFolder] = {
        "train": datasets.ImageFolder(root=str(train_dir), transform=tf["train"]),
        "val": datasets.ImageFolder(root=str(val_dir), transform=tf["val"]),
    }

    train_class_to_idx = image_datasets["train"].class_to_idx
    if image_datasets["val"].class_to_idx != train_class_to_idx:
        raise ValueError(
            f"val 類別索引與 train 不一致: "
            f"{image_datasets['val'].class_to_idx} != {train_class_to_idx}"
        )

    # [改進 #10] 驗證集過小時發出警告
    val_size = len(image_datasets["val"])
    if val_size < 100:
        warnings.warn(
            f"驗證集僅有 {val_size} 張圖片，指標可能不穩定。"
            f"建議從訓練集中額外劃分 10-15% 作為驗證集。"
        )

    idx_to_class = {v: k for k, v in train_class_to_idx.items()}
    return image_datasets, train_class_to_idx, idx_to_class


def _make_loader(
    ds: datasets.ImageFolder, batch_size: int, num_workers: int, shuffle: bool
) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=(shuffle),
    )


# [改進 #15] test dataset 延遲載入，不在訓練開始時就佔記憶體
def _build_test_loader(
    dataset_root: Path, img_size: int, batch_size: int, num_workers: int,
    expected_class_to_idx: Dict[str, int],
) -> DataLoader:
    tf = build_transforms(img_size=img_size)
    test_dir = dataset_root / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"資料夾不存在: {test_dir}")
    test_ds = datasets.ImageFolder(root=str(test_dir), transform=tf["test"])
    if test_ds.class_to_idx != expected_class_to_idx:
        raise ValueError(
            f"test 類別索引與 train 不一致: {test_ds.class_to_idx} != {expected_class_to_idx}"
        )
    return _make_loader(test_ds, batch_size, num_workers, shuffle=False)


# ── 類別權重 ─────────────────────────────────────────────────────────────────

# [改進 #1] 直接從 dataset.targets 計算，不需走 DataLoader 載入圖片
def compute_class_weights(
    dataset: datasets.ImageFolder, num_classes: int, device: torch.device
) -> torch.Tensor:
    targets = torch.tensor(dataset.targets, dtype=torch.long)
    counts = torch.bincount(targets, minlength=num_classes).float()
    counts = torch.clamp(counts, min=1.0)
    weights = counts.sum() / (num_classes * counts)
    logger.info("類別分佈: %s  權重: %s", counts.tolist(), weights.tolist())
    return weights.to(device)


# ── 模型 ─────────────────────────────────────────────────────────────────────

# [改進 #6] 分類頭加入 Dropout 降低過擬合
def build_model(
    model_name: str, num_classes: int, freeze_backbone: bool, dropout: float = 0.3
) -> nn.Module:
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
    else:
        raise ValueError(f"不支援的模型: {model_name}")

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )

    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False
    return model


# ── 指標計算 ─────────────────────────────────────────────────────────────────

def binary_metrics_from_logits(
    logits: torch.Tensor, labels: torch.Tensor, positive_index: int = 1
) -> Tuple[int, int, int, int]:
    preds = logits.argmax(dim=1)
    tp = ((preds == positive_index) & (labels == positive_index)).sum().item()
    tn = ((preds != positive_index) & (labels != positive_index)).sum().item()
    fp = ((preds == positive_index) & (labels != positive_index)).sum().item()
    fn = ((preds != positive_index) & (labels == positive_index)).sum().item()
    return tp, tn, fp, fn


# [改進 #16] 當樣本數為 0 時發出警告而非靜默回傳 0
def summarize_metrics(
    loss_sum: float, samples: int, tp: int, tn: int, fp: int, fn: int
) -> EpochResult:
    if samples == 0:
        warnings.warn("summarize_metrics 收到 0 個樣本，所有指標為 0。")
        return EpochResult(loss=0.0, accuracy=0.0, precision=0.0, recall=0.0, f1=0.0,
                           tp=0, tn=0, fp=0, fn=0)

    loss = loss_sum / samples
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return EpochResult(
        loss=loss, accuracy=accuracy, precision=precision, recall=recall, f1=f1,
        tp=tp, tn=tn, fp=fp, fn=fn,
    )


# ── 單 epoch 訓練 / 驗證 ────────────────────────────────────────────────────

# [改進 #2] AMP 混合精度  [改進 #7] 梯度裁剪
def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    positive_index: int = 1,
    scaler: Optional[GradScaler] = None,
    max_grad_norm: float = 1.0,
) -> EpochResult:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_samples = 0
    total_tp = total_tn = total_fp = total_fn = 0

    use_amp = scaler is not None and device.type == "cuda"

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            if is_train:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                    optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        tp, tn, fp, fn = binary_metrics_from_logits(
            logits.detach().float(), labels, positive_index=positive_index
        )
        total_tp += tp
        total_tn += tn
        total_fp += fp
        total_fn += fn

    return summarize_metrics(total_loss, total_samples, total_tp, total_tn, total_fp, total_fn)


# ── 主訓練流程 ───────────────────────────────────────────────────────────────

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
    deterministic: bool,
    early_stop_patience: int,
    dropout: float,
    backbone_lr_factor: float,
    max_grad_norm: float,
    resume_checkpoint: Optional[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(output_dir)

    set_seed(seed, deterministic=deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("使用裝置: %s", device)

    # ── 資料 ──────────────────────────────────────────────────────────────
    image_datasets, class_to_idx, idx_to_class = _build_datasets(dataset_root, img_size)
    num_classes = len(class_to_idx)
    if num_classes != 2:
        raise ValueError(f"此任務預期 2 類，但偵測到 {num_classes} 類: {class_to_idx}")

    positive_index = class_to_idx.get("PNEUMONIA", 1)
    logger.info("類別映射: %s, 正類別索引(PNEUMONIA): %d", class_to_idx, positive_index)

    train_loader = _make_loader(image_datasets["train"], batch_size, num_workers, shuffle=True)
    val_loader = _make_loader(image_datasets["val"], batch_size, num_workers, shuffle=False)

    # ── 模型 / 損失 / 優化器 ─────────────────────────────────────────────
    model = build_model(
        model_name=model_name, num_classes=num_classes,
        freeze_backbone=freeze_backbone, dropout=dropout,
    ).to(device)

    class_weights = compute_class_weights(image_datasets["train"], num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    start_epoch = 1
    best_val_f1 = -1.0
    epochs_no_improve = 0
    history: List[Dict[str, float]] = []

    # [改進 #13] 支援從 checkpoint 斷點續訓
    if resume_checkpoint and Path(resume_checkpoint).exists():
        ckpt = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_f1 = ckpt["best_val_f1"]
        history = ckpt.get("history", [])
        logger.info("從 checkpoint 恢復，繼續第 %d epoch", start_epoch)

    # ── 訓練迴圈 ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs + 1):

        # [改進 #5] 解凍時使用 add_param_group 保留 fc 層的 optimizer state
        if freeze_backbone and epoch == unfreeze_epoch:
            logger.info("解凍 backbone，進行全網路微調 (backbone lr=%.1e)...", lr * backbone_lr_factor)
            backbone_params = []
            for name, param in model.named_parameters():
                if not name.startswith("fc."):
                    param.requires_grad = True
                    backbone_params.append(param)
            optimizer.add_param_group({
                "params": backbone_params,
                "lr": lr * backbone_lr_factor,
            })

        train_metrics = run_one_epoch(
            model=model, loader=train_loader, criterion=criterion,
            optimizer=optimizer, device=device, positive_index=positive_index,
            scaler=scaler, max_grad_norm=max_grad_norm,
        )
        val_metrics = run_one_epoch(
            model=model, loader=val_loader, criterion=criterion,
            optimizer=None, device=device, positive_index=positive_index,
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

        logger.info(
            "[Epoch %02d/%d] Train loss=%.4f, acc=%.4f, f1=%.4f | "
            "Val loss=%.4f, acc=%.4f, f1=%.4f",
            epoch, epochs,
            train_metrics.loss, train_metrics.accuracy, train_metrics.f1,
            val_metrics.loss, val_metrics.accuracy, val_metrics.f1,
        )

        # [改進 #3] 不再 deepcopy，只存到磁碟
        if val_metrics.f1 > best_val_f1:
            best_val_f1 = val_metrics.f1
            torch.save(model.state_dict(), output_dir / "best_model.pth")
            logger.info("  ↑ 新最佳 val F1=%.4f，已儲存模型", best_val_f1)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # [改進 #13] 儲存完整 checkpoint（含 optimizer / scheduler / scaler）
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_val_f1": best_val_f1,
                "history": history,
            },
            output_dir / "last_checkpoint.pth",
        )

        # [改進 #4] Early Stopping
        if early_stop_patience > 0 and epochs_no_improve >= early_stop_patience:
            logger.info(
                "Early stopping: 連續 %d 個 epoch 無改善，於 epoch %d 停止訓練",
                early_stop_patience, epoch,
            )
            break

    # ── 載入最佳模型做測試 ────────────────────────────────────────────────
    best_path = output_dir / "best_model.pth"
    if not best_path.exists():
        raise RuntimeError("訓練未產生可用模型。")

    model.load_state_dict(torch.load(best_path, map_location=device))

    # [改進 #15] 延遲載入測試集
    test_loader = _build_test_loader(
        dataset_root, img_size, batch_size, num_workers,
        expected_class_to_idx=class_to_idx,
    )
    test_metrics = run_one_epoch(
        model=model, loader=test_loader, criterion=criterion,
        optimizer=None, device=device, positive_index=positive_index,
    )

    # [改進 #14] 輸出混淆矩陣
    logger.info(
        "\n[Test] loss=%.4f, acc=%.4f, precision=%.4f, recall=%.4f, f1=%.4f",
        test_metrics.loss, test_metrics.accuracy,
        test_metrics.precision, test_metrics.recall, test_metrics.f1,
    )
    logger.info(
        "[Test 混淆矩陣] TP=%d, TN=%d, FP=%d, FN=%d",
        test_metrics.tp, test_metrics.tn, test_metrics.fp, test_metrics.fn,
    )

    # ── 儲存紀錄 ─────────────────────────────────────────────────────────
    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    metadata = {
        "model_name": model_name,
        "class_to_idx": class_to_idx,
        "idx_to_class": {str(k): v for k, v in idx_to_class.items()},
        "img_size": img_size,
        "best_val_f1": best_val_f1,
        "test_metrics": {
            "loss": test_metrics.loss,
            "accuracy": test_metrics.accuracy,
            "precision": test_metrics.precision,
            "recall": test_metrics.recall,
            "f1": test_metrics.f1,
            "tp": test_metrics.tp,
            "tn": test_metrics.tn,
            "fp": test_metrics.fp,
            "fn": test_metrics.fn,
        },
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info("模型與紀錄已輸出到: %s", output_dir)


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chest X-ray Pneumonia Classification with ResNet Transfer Learning (Senior Edition)"
    )
    # [改進 #12] 使用相對路徑作為預設值
    default_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chest_xray")
    default_output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

    parser.add_argument("--dataset_root", type=str, default=default_root,
                        help="資料集根目錄，底下應包含 train/ val/ test")
    parser.add_argument("--output_dir", type=str, default=default_output)
    parser.add_argument("--model_name", type=str, choices=["resnet18", "resnet50"], default="resnet18")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Windows 建議先用 0，穩定後可調高")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="前幾個 epoch 先只訓練最後分類層")
    parser.add_argument("--unfreeze_epoch", type=int, default=4,
                        help="若啟用 freeze_backbone，於此 epoch 解凍 backbone")
    parser.add_argument("--deterministic", action="store_true",
                        help="啟用 cudnn deterministic 模式（較慢但可完全重現）")
    parser.add_argument("--early_stop_patience", type=int, default=5,
                        help="連續 N 個 epoch val F1 無改善則停止 (0=不啟用)")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="分類頭 Dropout 機率")
    parser.add_argument("--backbone_lr_factor", type=float, default=0.1,
                        help="解凍後 backbone 學習率 = lr * factor")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="梯度裁剪的最大 norm")
    parser.add_argument("--resume", type=str, default=None,
                        help="checkpoint 路徑，用於斷點續訓")
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
        deterministic=args.deterministic,
        early_stop_patience=args.early_stop_patience,
        dropout=args.dropout,
        backbone_lr_factor=args.backbone_lr_factor,
        max_grad_norm=args.max_grad_norm,
        resume_checkpoint=args.resume,
    )
