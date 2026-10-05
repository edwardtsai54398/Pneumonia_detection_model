import os
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from constant import IMAGENET_MEAN, IMAGENET_STD
from env import in_colab

KAGGLE_DATASET_ID = "paultimothymooney/chest-xray-pneumonia"


def ensure_kaggle_auth():
    """確保 kagglehub 有 Kaggle 憑證可用，回傳憑證來源字串。

    嘗試順序：
      1. 環境變數 KAGGLE_USERNAME / KAGGLE_KEY
      2. Colab Secrets（左側 🔑 圖示，新增同名兩個 secret 並開啟 Notebook access）
      3. ~/.kaggle/kaggle.json
      4. kagglehub.login() 互動式輸入

    憑證取得：Kaggle → 右上頭像 → Settings → API → Create New Token，
    下載的 kaggle.json 裡就是 username 與 key。
    """
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return "環境變數"

    if in_colab():
        try:
            from google.colab import userdata

            os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
            os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
            return "Colab Secrets"
        except Exception as e:
            print(f"未能從 Colab Secrets 取得憑證（{type(e).__name__}），改用互動式登入。")

    if (Path.home() / ".kaggle" / "kaggle.json").exists():
        return "~/.kaggle/kaggle.json"

    import kagglehub

    kagglehub.login()  # 會跳出輸入框要帳號與金鑰
    return "互動式登入"


def find_split_root(root):
    """在下載目錄底下找出真正含 train/val/test 的那一層。

    Kaggle 上這個資料集不同版本會多包一層 chest_xray/，所以不寫死路徑。
    """
    root = Path(root)
    candidates = [root, root / "chest_xray", root / "chest_xray" / "chest_xray"]
    candidates += sorted(root.glob("*/chest_xray"))

    for candidate in candidates:
        if all((candidate / split).is_dir() for split in ("train", "val", "test")):
            return candidate

    raise FileNotFoundError(f"在 {root} 底下找不到同時含 train/val/test 的資料夾")


def dataset_version(path):
    """從 kagglehub 快取路徑解析版本號（.../versions/N），解析不到回傳 None。

    Colab 的共用快取路徑（/kaggle/input/...）不含版本號，所以這個值可能是 None。
    """
    match = re.search(r"[\/]versions[\/](\d+)", str(path))
    return int(match.group(1)) if match else None


def download_dataset(local_dir=None):
    """下載 chest-xray-pneumonia，回傳含 train/val/test 的資料夾路徑。

    local_dir 指定且已有有效資料時，直接沿用不重新下載
    （例如事先解壓到 Drive 的副本）。
    """
    if local_dir is not None:
        try:
            split_root = find_split_root(local_dir)
            print(f"沿用既有資料：{split_root}")
            return split_root
        except FileNotFoundError:
            print(f"{local_dir} 沒有可用資料，改從 Kaggle 下載。")

    print(f"Kaggle 憑證來源：{ensure_kaggle_auth()}")

    import kagglehub

    cache_root = Path(kagglehub.dataset_download(KAGGLE_DATASET_ID))
    print(f"kagglehub 快取路徑：{cache_root}")
    return find_split_root(cache_root)


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
            transforms.Resize((256, 256)),
            transforms.CenterCrop(img_size),
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
            drop_last=(split == "train"),
        )
        for split in datasets
    }
