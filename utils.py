import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from constant import DEFAULT_MODEL_NAME
from engine.visualize import plot_cm, plot_history


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EpochTimer:
    def __init__(self):
        self._times = []
        self._last = time.time()

    def __call__(self, val_m, history, epoch):
        now = time.time()
        self._times.append(now - self._last)
        self._last = now

    def get_time_per_epoch(self):
        if not self._times:
            return 0.0
        avg = sum(self._times) / len(self._times)
        print(f"Average time per epoch: {avg:.1f}s")
        return avg


# 新增設定參數要同步加進這裡
CONFIG_KEYS = (
    "EXPERIMENT_NAME",
    "MODEL_NAME",
    "IMAGE_SIZE",
    "BATCH_SIZE",
    "EPOCHS",
    "LR",
    "WEIGHT_DECAY",
    "RANDOM_SEED",
    "SPLIT_SEED",
    "FREEZE_BACKBONE",
    "UNFREEZE_EPOCH",
    "BACKBONE_LR_FACTOR",
    "PATIENCE",
    "NUM_WORKERS",
    "TRAIN_RATIO",
    "VAL_RATIO",
    "TEST_RATIO",
    "GROUP_BY_PATIENT",
)


def collect_config(namespace, keys=CONFIG_KEYS):
    """從 notebook 的 globals() 收集設定，回傳可寫進 metadata.json 的 dict。

    放在設定 cell 的最後一行呼叫：少打或打錯一個變數名（EPOCHSS = 30）會在
    跑那個 cell 的當下就 KeyError，而不是訓練 40 分鐘之後才發現。
    """
    missing = [k for k in keys if k not in namespace]
    if missing:
        raise KeyError(f"設定 cell 缺少這些變數: {missing}")
    return {k.lower(): namespace[k] for k in keys}


def make_run_id(experiment_name):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{experiment_name}_{stamp}"


def make_output_dir(experiment_name, base="outputs", run_id=None):
    """建立並回傳這次實驗的輸出目錄。

    run_id 可以先用 make_run_id() 產生再傳進來。在 Colab 需要這樣做：權重寫到
    Drive、文字產物寫到 git clone 裡，兩邊要同一個資料夾名，而呼叫兩次
    make_output_dir() 會產生兩個不同的時間戳。
    """
    out = Path(base) / (run_id or make_run_id(experiment_name))
    out.mkdir(parents=True, exist_ok=True)
    return out


def _json_default(value):
    """json.dump 的保險絲：numpy 純量與 Path 都轉成原生型別。

    split_counts() 已經轉過 int，但設定 cell 裡隨手放一個 np.float64 就會讓
    整個 save_results 在最後一步炸掉——那時候訓練已經跑完了。
    """
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def save_results(
    out_dir,
    model,
    best_state,
    best_val_f1,
    best_epoch,
    test_metrics,
    class_to_idx,
    idx_to_class,
    img_size,
    history,
    model_name=DEFAULT_MODEL_NAME,
    train_time_per_epoch=0.0,
    config=None,
    run_info=None,
    split_info=None,
    weights_dir=None,
):
    model.load_state_dict(best_state)

    num_parameters = sum(p.numel() for p in model.parameters()) / 1e6  # 單位：M

    # 權重可另存，文字產物留在 out_dir
    weights_dir = Path(weights_dir) if weights_dir is not None else out_dir
    weights_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, weights_dir / "best_model.pth")

    metadata = {
        "model_name": model_name,
        "num_parameters": num_parameters,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "img_size": img_size,
        "best_val_f1": best_val_f1,
        # 0-based，對齊 history
        "best_epoch": best_epoch,
        "test_metrics": {k: v for k, v in test_metrics.items() if k != "cm"},
        "train_time_per_epoch": train_time_per_epoch,
    }
    # 可重現性欄位
    if config is not None:
        metadata["config"] = dict(config)
    if run_info is not None:
        metadata.update(run_info)
    if split_info is not None:
        metadata.update(split_info)

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, default=_json_default)

    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2, default=_json_default)

    history_fig = plot_history(history)
    history_fig.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close(history_fig)

    if test_metrics.get("cm") is not None:
        class_names = [idx_to_class[0], idx_to_class[1]]
        cm_fig = plot_cm(test_metrics["cm"], class_names=class_names)
        cm_fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
        plt.close(cm_fig)

    print(f"Results saved to {out_dir}")
    if weights_dir != out_dir:
        print(f"Weights saved to {weights_dir}")
