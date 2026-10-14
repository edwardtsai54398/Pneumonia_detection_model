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
from data.eda import mode_counts, size_summary
from engine.visualize import (
    plot_aspect_hist,
    plot_class_distribution,
    plot_cm,
    plot_history,
    plot_sample_grid,
    plot_size_scatter,
    plot_transform_comparison,
)


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


def save_eda_results(
    out_dir,
    counts,
    counts_after,
    survey,
    group_stats=None,
    run_info=None,
    img_size=224,
    n_samples_per_class=4,
    seed=0,
    sample_paths=None,
):
    """把 EDA 的統計與圖一次存進 out_dir。

    與 save_results 的分工：那一支存的是「某一次訓練」的產物，所以目錄帶時間
    戳；這一支存的是「資料集本身」的分析，與 run 無關，重跑就覆蓋同一份。

    參數裡的 counts / counts_after / survey 直接吃 count_images()、
    split_counts()、survey_images() 的回傳值，不需要先整理。
    group_stats 是 group_counts() 的回傳值，沒開分組切分就留 None。

    sample_paths 是前後對照圖要用的影像。呼叫端若已經在畫面上顯示過一張對照
    圖，就把同一份清單傳進來——否則這裡會自己挑，存檔的圖跟剛剛看到的那張不
    是同幾張，報告引用時會對不上。留 None 時每類取第一張（排序後固定，不隨機）。

    沒有回傳值。寫出的檔案：
      eda_summary.json              所有數字，含 size_summary / mode_counts
      original_split_distribution.png
      regrouped_split_distribution.png
      color_mode_distribution.png
      image_size_scatter.png
      aspect_ratio_hist.png
      sample_grid.png
      transform_comparison.png
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sizes = size_summary(survey)
    modes = mode_counts(survey)

    summary = {
        "counts": counts,
        "counts_after": counts_after,
        "size_summary": sizes,
        "mode_counts": modes,
    }
    if group_stats is not None:
        summary["group_counts"] = group_stats
    if run_info is not None:
        summary.update(run_info)

    with open(out_dir / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=_json_default)

    if sample_paths is None:
        # 每類取第一張做前後對照
        sample_paths = [
            records[0]["path"] for records in survey.values() if records
        ]

    figures = {
        "original_split_distribution.png": plot_class_distribution(
            counts, title="Original Kaggle Split"
        ),
        "regrouped_split_distribution.png": plot_class_distribution(
            counts_after, title="Regrouped 70/15/15"
        ),
        "color_mode_distribution.png": plot_class_distribution(
            modes, title="Color Mode by Class"
        ),
        "image_size_scatter.png": plot_size_scatter(survey),
        "aspect_ratio_hist.png": plot_aspect_hist(survey),
        "sample_grid.png": plot_sample_grid(
            survey, n_per_class=n_samples_per_class, seed=seed
        ),
        "transform_comparison.png": plot_transform_comparison(
            sample_paths, img_size=img_size, seed=seed
        ),
    }

    for filename, fig in figures.items():
        fig.savefig(out_dir / filename, dpi=150)
        plt.close(fig)

    print(f"EDA results saved to {out_dir}")
