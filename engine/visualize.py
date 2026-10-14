import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image

from constant import IMAGENET_MEAN, IMAGENET_STD
from data.dataset import build_transforms

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


def plot_size_scatter(survey, show=False):
    """Plot and return a width-vs-height scatter of the raw images.

    吃 data.eda.survey_images() 的回傳值。每個類別一個顏色、一個 marker。

    alpha 壓到 0.25 是因為這份資料集有大量重複尺寸：不透明的話幾千個點會疊成
    一塊實心色塊，看不出密度集中在哪。再畫一條 y=x 虛線當參考，點落在線上就是
    正方形，肉眼可以直接判斷偏離程度。
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    all_values = []
    for i, (class_name, records) in enumerate(survey.items()):
        widths = [r["width"] for r in records]
        heights = [r["height"] for r in records]
        all_values += widths + heights
        ax.scatter(
            widths,
            heights,
            s=12,
            alpha=0.25,
            linewidths=0,
            color=CLASS_COLORS[i % len(CLASS_COLORS)],
            label=f"{class_name} (n={len(records)})",
        )

    if all_values:
        # y=x 參考線標出正方形
        lim = [0, max(all_values) * 1.05]
        ax.plot(lim, lim, linestyle="--", linewidth=1, color="gray", zorder=0)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
    # 等比才讓 y=x 真的是 45 度
    ax.set_aspect("equal")

    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.set_title("Raw Image Size Distribution", fontsize=14)
    # 圖例的點不透明才看得清
    legend = ax.legend(frameon=False)
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)

    ax.grid(linewidth=0.5, alpha=0.3)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_aspect_hist(survey, bins=40, show=False):
    """Plot and return a histogram of width/height ratios.

    吃 data.eda.survey_images() 的回傳值。兩個類別疊在同一軸上比較。

    共用同一組 bin 邊界（而不是各自讓 matplotlib 自動切），否則兩個類別的長條
    寬度不一致，疊起來的高度就沒有可比性。
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    per_class = {
        name: [r["width"] / r["height"] for r in records]
        for name, records in survey.items()
    }
    flat = [a for values in per_class.values() for a in values]
    # 兩類共用 bin 才可比
    edges = np.histogram_bin_edges(flat, bins=bins) if flat else bins

    for i, (class_name, aspects) in enumerate(per_class.items()):
        ax.hist(
            aspects,
            bins=edges,
            alpha=0.6,
            color=CLASS_COLORS[i % len(CLASS_COLORS)],
            label=class_name,
        )

    # 1.0 是正方形
    ax.axvline(1.0, linestyle="--", linewidth=1, color="gray", zorder=0)

    ax.set_xlabel("Aspect Ratio (width / height)")
    ax.set_ylabel("Number of Images")
    ax.set_title("Aspect Ratio Distribution", fontsize=14)
    ax.legend(frameon=False)

    # 只留 y 方向的細實線格線
    ax.grid(axis="y", linewidth=0.5, alpha=0.3)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_sample_grid(survey, n_per_class=4, seed=0, show=False):
    """Plot and return a grid of raw sample images, one row per class.

    吃 data.eda.survey_images() 的回傳值——用 survey 而不是單純的路徑清單，是
    因為每一格的標題要標上該張的原始尺寸與色彩模式，那兩個值 survey 已經有了，
    不必再開一次檔。

    固定 seed 抽樣，重跑產生同一張圖，報告裡的圖才不會每次都換。
    灰階圖用 cmap="gray" 畫；RGB 的直接畫，matplotlib 會忽略 cmap。
    """
    rng = np.random.default_rng(seed)

    n_rows = len(survey)
    fig, axes = plt.subplots(
        n_rows, n_per_class, figsize=(3 * n_per_class, 3.4 * n_rows), squeeze=False
    )

    for row, (class_name, records) in enumerate(survey.items()):
        picks = rng.choice(len(records), size=min(n_per_class, len(records)), replace=False)
        for col in range(n_per_class):
            ax = axes[row][col]
            ax.set_xticks([])
            ax.set_yticks([])
            # 置頂才讓標題對齊
            ax.set_anchor("N")
            if col >= len(picks):
                ax.axis("off")
                continue

            record = records[picks[col]]
            with Image.open(record["path"]) as img:
                ax.imshow(img, cmap="gray")
            ax.set_title(
                f"{record['width']}x{record['height']}  {record['mode']}", fontsize=9
            )
            if col == 0:
                ax.set_ylabel(class_name, fontsize=12)

    fig.suptitle("Raw Samples", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if show:
        plt.show()
    return fig


def _denormalize(tensor):
    """把正規化過的 (3, H, W) tensor 還原成可顯示的 (H, W, 3) 陣列。

    build_transforms 的輸出減過 ImageNet 均值、除過標準差，直接 imshow 會是一
    片過曝或死黑。這裡乘回標準差、加回均值，再 clip 到 [0, 1]——clip 是必要的：
    RandomResizedCrop 之後的極值反推回來會略微超出 [0, 1]，imshow 遇到超範圍的
    float 會整張警告並自行截斷。

    回傳 numpy float 陣列，形狀 (H, W, 3)，值域 [0, 1]。
    """
    array = tensor.numpy().transpose(1, 2, 0)
    array = array * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return np.clip(array, 0.0, 1.0)


def plot_transform_comparison(paths, img_size=224, seed=0, show=False):
    """Plot and return a before/after grid of the preprocessing pipeline.

    每一列是一張圖，四個欄位依序是：
      Original        原始檔，尺寸與色彩模式都未經處理
      Eval transform  Grayscale(3) + Resize(256) + CenterCrop，確定性的
      Train aug #1/#2 同一張圖跑兩次 train transform，用來顯示隨機增強的幅度

    刻意重用 data.dataset.build_transforms 而不是在這裡重寫一套 transform：
    這張圖的用途就是展示「實際餵進模型的東西長什麼樣」，自己另寫一份就失去意義
    （而且兩邊遲早會分歧）。

    torch.manual_seed 固定隨機增強，報告裡的圖重跑才一致。
    """
    tf = build_transforms(img_size=img_size)
    columns = ["Original", "Eval transform", "Train aug #1", "Train aug #2"]

    paths = list(paths)
    fig, axes = plt.subplots(
        len(paths), len(columns), figsize=(3 * len(columns), 3.4 * len(paths)),
        squeeze=False,
    )

    torch.manual_seed(seed)
    for row, path in enumerate(paths):
        with Image.open(path) as img:
            original = img.convert("RGB")
            size_label = f"{img.width}x{img.height} {img.mode}"

        rendered = [
            np.asarray(original),
            _denormalize(tf["val"](original)),
            _denormalize(tf["train"](original)),
            _denormalize(tf["train"](original)),
        ]

        for col, (ax, image) in enumerate(zip(axes[row], rendered)):
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
            # 置頂才讓標題對齊
            ax.set_anchor("N")
            if row == 0:
                ax.set_title(columns[col], fontsize=11)
            if col == 0:
                ax.set_ylabel(size_label, fontsize=9)

    fig.suptitle("Preprocessing: Before vs After", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if show:
        plt.show()
    return fig
