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


def make_output_dir(experiment_name, base="outputs"):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(base) / f"{experiment_name}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_results(
    out_dir,
    model,
    best_state,
    best_val_f1,
    test_metrics,
    class_to_idx,
    idx_to_class,
    img_size,
    history,
    model_name=DEFAULT_MODEL_NAME,
    train_time_per_epoch=0.0,
):
    model.load_state_dict(best_state)

    num_parameters = sum(p.numel() for p in model.parameters()) / 1e6  # 單位：M

    torch.save(best_state, out_dir / "best_model.pth")

    metadata = {
        "model_name": model_name,
        "num_parameters": num_parameters,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "img_size": img_size,
        "best_val_f1": best_val_f1,
        "test_metrics": {k: v for k, v in test_metrics.items() if k != "cm"},
        "train_time_per_epoch": train_time_per_epoch,
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    history_fig = plot_history(history)
    history_fig.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close(history_fig)

    if test_metrics.get("cm") is not None:
        class_names = [idx_to_class[0], idx_to_class[1]]
        cm_fig = plot_cm(test_metrics["cm"], class_names=class_names)
        cm_fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
        plt.close(cm_fig)

    print(f"Results saved to {out_dir}")
