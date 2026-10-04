import matplotlib
import numpy as np
import matplotlib.pyplot as plt

# 固定配色，不依大小變色
CLASS_COLORS = ["#1f77b4", "#ff7f0e"]


def plot_history(history, show=False):
    """Plot and return a 2×3 figure of training curves."""
    epochs = [h["epoch"] for h in history]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    pairs = [
        (axes[0, 0], "Loss", "train_loss", "val_loss"),
        (axes[0, 1], "F1 Score", "train_f1", "val_f1"),
        (axes[0, 2], "Accuracy", "train_acc", "val_acc"),
        (axes[1, 0], "Precision", "train_precision", "val_precision"),
        (axes[1, 1], "Recall", "train_recall", "val_recall"),
    ]
    for ax, title, train_key, val_key in pairs:
        ax.plot(epochs, [h[train_key] for h in history], marker="o", label="Train")
        ax.plot(epochs, [h[val_key] for h in history], marker="s", label="Val")
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True)

    ax = axes[1, 2]
    ax.plot(epochs, [h["lr"] for h in history], marker="D", color="tab:purple", label="LR")
    ax.set_title("Learning Rate", fontsize=14)
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True)

    for row in axes:
        for a in row:
            a.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    fig.suptitle("Training History", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if show:
        plt.show()
    return fig


def plot_cm(cm, class_names=None, show=False):
    """Plot and return a row-normalised confusion matrix figure."""
    cm = np.asarray(cm)
    num_classes = cm.shape[0]
    labels = class_names if class_names is not None else list(range(num_classes))

    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        cm_norm,
        interpolation="nearest",
        cmap=matplotlib.colormaps["Blues"],
        vmin=0,
        vmax=1,
    )
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted Label",
        ylabel="True Label",
        title="Confusion Matrix (Row-Normalised)",
    )

    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j, i,
                f"{cm_norm[i, j]:.2f}\n({cm[i, j]})",
                ha="center", va="center", fontsize=9,
                color="white" if cm_norm[i, j] > 0.5 else "black",
            )

    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_class_distribution(counts, title="Class Distribution", show=False):
    """Plot and return a grouped bar chart of class counts.

    接受兩種形狀：
      {"NORMAL": 1583, "PNEUMONIA": 4273}                  -> 單一組
      {"train": {"NORMAL": 1341, ...}, "val": {...}, ...}  -> 每個 split 一組

    用 matplotlib 而不是 seaborn，是為了不替這一張圖多引入 seaborn + pandas
    兩個依賴（而且 seaborn 的 palette 沒搭配 hue 已經被標記為 deprecated）。
    """
    nested = counts if all(isinstance(v, dict) for v in counts.values()) else {"all": counts}

    groups = list(nested)
    class_names = list(dict.fromkeys(name for g in nested.values() for name in g))

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(groups), dtype=float)
    n_series = len(class_names)
    # 總寬 0.76，系列間留 0.04 空隙
    gap = 0.04
    width = (0.76 - gap * (n_series - 1)) / n_series

    for i, name in enumerate(class_names):
        offset = (i - (n_series - 1) / 2) * (width + gap)
        values = [nested[g].get(name, 0) for g in groups]
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=name,
            color=CLASS_COLORS[i % len(CLASS_COLORS)],
        )
        # 每根都標數值，補足對比不足
        ax.bar_label(bars, padding=2, fontsize=9)

    ax.set_xticks(x, groups)
    ax.set_ylabel("Number of Images")
    ax.set_title(title, fontsize=14)
    ax.legend(frameon=False)

    # 只留 y 方向的細實線格線
    ax.grid(axis="y", linewidth=0.5, alpha=0.3)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    # 留空間給最高的數值標籤
    ax.margins(y=0.12)

    fig.tight_layout()
    if show:
        plt.show()
    return fig
