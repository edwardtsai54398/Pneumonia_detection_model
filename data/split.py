"""索引式的資料切分：只產生「哪張圖屬於哪個 split」的名單，不建立實體資料夾。

為什麼不建 symlink 樹：原本的做法是在 /content/preprocessedData 真的建出
train/val/test 資料夾並為 5856 張圖各建一個 symlink，再交給 ImageFolder 掃。
那個做法的缺陷幾乎都來自「建立檔案」這個動作本身——檔名撞到會 FileExistsError
（而且是在 rmtree 砍掉舊資料之後才爆）、symlink 指向唯讀快取所以 runtime 重啟
後全部失效（而 ImageFolder 掃目錄時不會發現，要等 epoch 中途 PIL 去讀才爆）、
Windows 建 symlink 需要額外權限。不建資料夾，這些問題就不存在，而且切分變成
可以 hash 並寫進 metadata.json 的資料結構，不是得去信任的檔案系統狀態。
"""

import hashlib
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

from constant import CLASS_NAMES, DEFAULT_SPLIT_SEED
from data.dataset import build_transforms

SPLITS = ["train", "val", "test"]
IMG_EXTS = {".jpeg", ".jpg", ".png"}

# 從 CLASS_NAMES 推導，不是從觀察到的資料推導。這很重要：如果某個 split 剛好
# 缺了一個類別，從資料推導會產生不連續的索引，而 infer_single.py 是用
# `for i in range(num_classes): idx_to_class[i]` 讀的，缺一格就 KeyError。
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}


def count_images(root):
    """清點原始資料集每個 split / 類別的張數，回傳 {split: {class: count}}。

    不寫死副檔名（原本的 glob("*.jpeg") 換個資料集就會對不上），並且過濾掉
    非目錄的項目，免得 split 資料夾裡有 .DS_Store 之類的東西就壞掉。
    """
    root = Path(root)
    counts = {}
    for split in SPLITS:
        split_dir = root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"資料夾不存在: {split_dir}")
        counts[split] = {
            d.name: sum(1 for p in d.iterdir() if p.is_file())
            for d in sorted(split_dir.iterdir())
            if d.is_dir()
        }
    return counts


def collect_all_images(root):
    """把原始 train/val/test 的圖全部收集起來，依類別分組。

    回傳 {"NORMAL": [Path, ...], "PNEUMONIA": [Path, ...]}，兩個清單都已排序，
    所以同一份資料集每次收集的順序都一樣（切分的可重現性靠這個）。

    刻意不做「非 NORMAL 就當 PNEUMONIA」的歸類：遇到 CLASS_NAMES 以外的目錄名
    直接 raise，寧可吵一下也不要把不明資料默默塞進某一類。
    """
    root = Path(root)
    collected = {name: [] for name in CLASS_NAMES}

    for split in SPLITS:
        split_dir = root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"資料夾不存在: {split_dir}")
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            if class_dir.name not in CLASS_TO_IDX:
                raise ValueError(
                    f"預期外的類別目錄 {class_dir}；CLASS_NAMES = {CLASS_NAMES}"
                )
            for file in sorted(class_dir.iterdir()):
                if file.suffix.lower() in IMG_EXTS:
                    collected[class_dir.name].append(file)

    # 全域排序，讓結果不依賴走訪順序。只在每個目錄內排序的話，整份清單的順序
    # 會跟著 SPLITS 的順序走——哪天有人改了那個順序，切分結果就會跟著變，而
    # 那是無聲的：fingerprint 會變但沒人知道為什麼。
    for name in collected:
        collected[name].sort()

    return collected


def _allocate(total, ratios):
    """把 total 依 ratios 分配成整數，且保證加總等於 total（最大餘額法）。

    單純對每一份取 int() 會把餘數整個丟掉——1583 張配 .7/.15/.15 會變成
    1108+237+237=1582，少一張。這裡把餘數補給小數部分最大的那幾份。
    """
    exact = [total * r for r in ratios]
    base = [int(x) for x in exact]
    remainder = total - sum(base)
    if remainder:
        order = sorted(
            range(len(ratios)), key=lambda i: exact[i] - base[i], reverse=True
        )
        for i in order[:remainder]:
            base[i] += 1
    return base


def plan_splits(
    all_images,
    *,
    split_seed=DEFAULT_SPLIT_SEED,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    group_key=None,
):
    """依比例切出 train/val/test 名單，每個類別各自切分（stratified）。

    回傳 {"train": [(Path, class_idx), ...], "val": [...], "test": [...]}。

    參數刻意叫 split_seed 而不是 seed：這樣把訓練用的種子誤傳進來
    （plan_splits(..., seed=RANDOM_SEED)）會直接 TypeError，而不是無聲地
    讓「換 seed 比較穩定性」同時換掉整個資料切分。

    group_key 預設 None（純按檔案切分）。給一個 Path -> 群組鍵 的函式時會改成
    整組一起進同一個 split，用來避免同一位病人的影像同時出現在 train 和 test。
    這會讓實際張數不再精確等於比例（比例是套在群組數上）。
    """
    ratios = (train_ratio, val_ratio, test_ratio)
    if any(r < 0 for r in ratios):
        raise ValueError(f"比例不可為負: {ratios}")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(
            f"train/val/test 比例必須加總為 1，目前 {ratios} 加總 {sum(ratios)}"
        )

    rng = np.random.default_rng(split_seed)
    plan = {split: [] for split in SPLITS}

    for class_name in CLASS_NAMES:
        files = all_images.get(class_name, [])
        class_idx = CLASS_TO_IDX[class_name]

        # 切分單位：預設是單張圖，給了 group_key 就是一整個群組
        if group_key is None:
            units = [[f] for f in files]
        else:
            groups = {}
            for f in files:
                groups.setdefault(group_key(f), []).append(f)
            units = [groups[k] for k in sorted(groups)]

        order = rng.permutation(len(units))
        counts = _allocate(len(units), ratios)

        start = 0
        for split, n in zip(SPLITS, counts):
            for i in order[start : start + n]:
                plan[split].extend((f, class_idx) for f in units[i])
            start += n

    # 把每個 split 內部打散，這樣即使之後有人用 shuffle=False 也不會拿到
    # 「先全部 NORMAL 再全部 PNEUMONIA」的順序。
    for split in SPLITS:
        samples = plan[split]
        plan[split] = [samples[i] for i in rng.permutation(len(samples))]

    return plan


def split_counts(plan):
    """每個 split 的類別分佈，{split: {class_name: count}}，值都是 int（可 JSON 化）。"""
    return {
        split: {
            name: int(sum(1 for _, idx in samples if idx == CLASS_TO_IDX[name]))
            for name in CLASS_NAMES
        }
        for split, samples in plan.items()
    }


def _relative_key(path):
    """用原始樹裡的 split/class/filename 當識別碼，跨機器穩定。

    不用完整路徑，因為 Colab 是 /kaggle/input/... 而本機是別的位置，
    否則同一個切分在兩台機器上會算出不同的 fingerprint。
    """
    return "/".join(Path(path).parts[-3:])


def split_fingerprint(plan):
    """切分內容的 sha256。兩次實驗要能互相比較，這個值必須相同。

    這是讓「seed-43 的 test 有 172/238 張在 seed-42 的 train 裡」這種事
    事後從兩份 metadata.json 就看得出來的東西。
    """
    h = hashlib.sha256()
    for split in SPLITS:
        h.update(split.encode())
        for key in sorted(f"{_relative_key(p)}:{idx}" for p, idx in plan[split]):
            h.update(key.encode())
    return h.hexdigest()


class SampleListDataset(Dataset):
    """用一份 [(path, class_idx)] 名單當資料集，行為對齊 ImageFolder。

    暴露 .targets / .classes / .class_to_idx，因為 compute_class_weights 讀
    dataset.targets，而 make_dataloaders 本來就是泛型的——所以 trainer 那邊
    完全不用改。
    """

    def __init__(
        self, samples, transform=None, class_to_idx=None, loader=default_loader
    ):
        self.samples = [(Path(p), int(idx)) for p, idx in samples]
        self.transform = transform
        self.loader = loader
        self.class_to_idx = dict(
            class_to_idx if class_to_idx is not None else CLASS_TO_IDX
        )
        self.classes = [
            name for name, _ in sorted(self.class_to_idx.items(), key=lambda kv: kv[1])
        ]
        self.targets = [idx for _, idx in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self.loader(str(path))
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def _verify_paths(samples, split):
    """確認名單裡的每個路徑都還在。

    主要擋的是「拿著上一個 runtime 算好的 plan 繼續用」——那些路徑指向已經
    消失的快取。在這裡失敗很大聲，而不是訓練到第三個 epoch 才在 PIL 裡爆。

    刻意檢查全部而不是抽樣：重切之後每個 split 的檔案來自原始的三個 split，
    所以「原始資料少了一部分」的情況下，抽樣很容易整批漏掉（實測抽 20 筆會
    漏掉只佔 8/45 的那批）。5856 次 stat 在訓練前跑一次的成本可以忽略。
    """
    if not samples:
        raise ValueError(f"{split} split 是空的")
    missing = [path for path, _ in samples if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"{split} split 有 {len(missing)} / {len(samples)} 個檔案不存在，"
            f"例如: {missing[0]}\n"
            "資料集快取可能已經失效（例如換過執行階段）。請重跑下載與切分的 cell。"
        )


def build_datasets_from_plan(plan, img_size=224, verify=True):
    """把 plan 變成 train/val/test 的 Dataset。

    回傳與 data.dataset.build_datasets 相同的三元組
    (image_datasets, class_to_idx, idx_to_class)，所以 train_model /
    make_dataloaders / save_results 都不用改。
    """
    tf = build_transforms(img_size=img_size)

    image_datasets = {}
    for split in SPLITS:
        samples = plan[split]
        if verify:
            _verify_paths(samples, split)
        image_datasets[split] = SampleListDataset(
            samples, transform=tf[split], class_to_idx=CLASS_TO_IDX
        )

    return image_datasets, dict(CLASS_TO_IDX), dict(IDX_TO_CLASS)
