"""
索引式的資料切分：只產生「哪張圖屬於哪個 split」的名單，不建立實體資料夾。
"""

import re
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

from constant import CLASS_NAMES, DEFAULT_SPLIT_SEED
from data.dataset import build_transforms

SPLITS = ["train", "val", "test"]
IMG_EXTS = {".jpeg", ".jpg", ".png"}

# 由 CLASS_NAMES 推導
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}


def count_images(root):
    """清點原始資料集每個 split / 類別的張數。

    不寫死副檔名（原本的 glob("*.jpeg") 換個資料集就會對不上），並且過濾掉
    非目錄的項目，免得 split 資料夾裡有 .DS_Store 之類的東西就壞掉。

    回傳 {split: {class_name: count}}
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

    刻意不做「非 NORMAL 就當 PNEUMONIA」的歸類：遇到 CLASS_NAMES 以外的目錄名
    直接 raise，寧可吵一下也不要把不明資料默默塞進某一類。

    回傳 {class_name: [Path, ...]}，例如 {"NORMAL": [...], "PNEUMONIA": [...]}
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

    for name in collected:
        collected[name].sort()

    return collected


def _allocate(total, ratios):
    """把 total 依 ratios 分配成整數，且保證加總等於 total（最大餘額法）。

    單純對每一份取 int() 會把餘數整個丟掉——1583 張配 .7/.15/.15 會變成
    1108+237+237=1582，少一張。這裡把餘數補給小數部分最大的那幾份。

    回傳 [count, ...]，長度與 ratios 相同、順序與 ratios 一一對應（呼叫端
    傳的是 (train, val, test)，所以第 0/1/2 個就是 train/val/test 的份額）。
    每個 count 是「切分單位」的個數而不是張數
    加總保證等於 total。
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


def patient_group_key(path):
    """從檔名取出病人識別碼，給 plan_splits(group_key=...) 用。

    這份資料集的 PNEUMONIA 檔名帶病人編號，NORMAL 不帶：

        train/PNEUMONIA/person1_bacteria_1.jpeg  -> 'train/PNEUMONIA/person1'
        train/PNEUMONIA/person1_virus_6.jpeg     -> 'train/PNEUMONIA/person1'
        train/NORMAL/IM-0115-0001.jpeg           -> 'train/NORMAL/IM-0115-0001'

    為什麼鍵裡要含原始的 split 與類別：collect_all_images 會把原始 train/val/test
    三棵樹合併成一份清單，而原始 train 和原始 test 底下都存在 person1_*，它們
    是不是同一個人光看檔名無法確定。只用 'person1' 當鍵會把兩批可能無關的影像
    硬綁成一組——不會報錯，只是分組變得沒有意義。寧可把同一人誤拆成兩組（退化
    成接近逐張切分），也不要把兩個人誤併成一組（那會讓切分的統計假設失效）。

    NORMAL 沒有編號可抽，就讓每張圖自己成為一組。

    回傳一個字串形如 "{原始 split}/{類別目錄名}/{病人編號或檔名 stem}"
    """
    parts = Path(path).parts[-3:]
    if len(parts) < 3:
        raise ValueError(f"路徑層級不足，無法取出 split/class/filename: {path}")
    split, class_name, filename = parts
    match = re.match(r"(person\d+)_", filename)
    stem = match.group(1) if match else Path(filename).stem
    return f"{split}/{class_name}/{stem}"


def _allocate_groups(units, ratios, rng):
    """把群組分配到各 split，讓「張數」盡量接近 ratios。

    做法是 LPT（longest processing time first）：大組先放，每次放進「離自己的
    張數配額還差最多」的那個 split。大組必須先放——小組留到後面才有東西可以
    填補誤差；反過來先放小組，最後那個大組不管丟哪都會把該 split 撐爆。

    回傳 [[unit_idx, ...], ...]：
      外層  長度與順序對應 SPLITS
      內層  units 的「索引」，不是 Path 也不是張數——呼叫端要用 units[i] 才拿
            得到那一組的檔案清單。
    """
    quota = [sum(len(u) for u in units) * r for r in ratios]

    order = sorted(
        rng.permutation(len(units)), key=lambda i: len(units[i]), reverse=True
    )

    buckets = [[] for _ in SPLITS]
    assigned = [0] * len(SPLITS)
    for i in order:
        s = max(range(len(SPLITS)), key=lambda s: quota[s] - assigned[s])
        buckets[s].append(i)
        assigned[s] += len(units[i])
    return buckets


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

    回傳 {split: [(path, class_idx), ...]}

    group_key 預設 None（純按檔案切分）。給一個 Path 回傳 群組鍵(str) t，用來避免同一位病人的影像同時出現在 train 和 test。
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

        # 切分單位：單張圖或一個群組
        if group_key is None:
            units = [[f] for f in files]
        else:
            groups = {}
            for f in files:
                groups.setdefault(group_key(f), []).append(f)
            units = [groups[k] for k in sorted(groups)]

        if group_key is None:
            order = rng.permutation(len(units))
            counts = _allocate(len(units), ratios)
            buckets = []
            start = 0
            for n in counts:
                buckets.append(order[start : start + n])
                start += n
        else:
            buckets = _allocate_groups(units, ratios, rng)

        for split, bucket in zip(SPLITS, buckets):
            for i in bucket:
                plan[split].extend((f, class_idx) for f in units[i])

    for split in SPLITS:
        samples = plan[split]
        plan[split] = [samples[i] for i in rng.permutation(len(samples))]

    return plan


def split_counts(plan):
    """每個 split 的類別分佈。

    回傳 {split: {class_name: count}}：
      split       直接沿用 plan 的鍵，也就是重切後的 "train"/"val"/"test"。
      class_name  固定列出 CLASS_NAMES 的每一個類別，就算該 split 一張都沒有
                  也會有這一格、值為 0——這樣「某個 split 掉了一整個類別」
                  在印出來的表上是看得見的 0，而不是消失的一列。
      count       張數，刻意轉成 python int 而不是 np.int64，才能直接
                  json.dump 進 metadata.json。
    """
    return {
        split: {
            name: int(sum(1 for _, idx in samples if idx == CLASS_TO_IDX[name]))
            for name in CLASS_NAMES
        }
        for split, samples in plan.items()
    }


def group_counts(all_images, group_key):
    """每個類別的群組數。

    用來確認 group_key 真的抓到了病人編號：如果群組數等於張數，表示沒有任何
    檔名被分到同一組（正規表示式沒對上），分組切分就等於沒開。

    回傳 {class_name: count}：
      count       該類別底下相異 group_key 的數量，也就是切分單位的個數，
                  不是張數。
    """
    return {
        name: len({group_key(f) for f in all_images.get(name, [])})
        for name in CLASS_NAMES
    }


class SampleListDataset(Dataset):
    """用一份 [(path, class_idx)] 名單當資料集，行為對齊 ImageFolder。

    暴露 .targets / .classes / .class_to_idx，因為 compute_class_weights 讀
    dataset.targets，而 make_dataloaders 本來就是泛型的——所以 trainer 那邊
    完全不用改。

    對外的資料形狀（刻意與 ImageFolder 同義，下游才能互換）：
      .samples        [(Path, class_idx), ...]，順序即 index 的順序；路徑一律
                      正規化成 Path、類別一律 int，不管傳進來的是 str 還是
                      np.int64。
      .targets        [class_idx, ...]，與 .samples 同順序、同長度。只有標籤
                      沒有路徑，給 compute_class_weights 數類別分佈用。
      .class_to_idx   {class_name: idx}
      .classes        [class_name, ...]，依 idx 由小到大排列，所以
                      classes[idx] 就是該索引的類別名。
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
        """樣本張數（int），也就是 index 的合法範圍是 0 .. len-1。"""
        return len(self.samples)

    def __getitem__(self, index):
        """回傳一筆 (image, target)，與 ImageFolder 的單筆格式相同。

        image   套過 transform 之後的結果。配 build_transforms 時是形狀
                (3, img_size, img_size)、已做 ImageNet 正規化的 float tensor
                （灰階 X 光被複製成 3 通道）；transform=None 時則是 loader
                吐出的 PIL RGB Image，沒有縮放也沒有正規化。
        target  int，class_idx 本身（不是 one-hot、不是類別名），可直接餵給
                CrossEntropyLoss。
        """
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

    沒有回傳值（None）——這是個純斷言：不是空清單、而且每個路徑都存在時就
    靜靜返回，否則 raise。回傳 None 不代表「沒問題但也沒檢查」。
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
    make_dataloaders / save_results 都不用改：
      image_datasets  {split: SampleListDataset}，三個 split 的鍵都在，各自
                      已套好對應的 transform——train 那份含隨機增強（所以同
                      一個 index 每次取到的 tensor 不同），val/test 那份只有
                      縮放與正規化，是確定性的。
      class_to_idx    {class_name: idx}
      idx_to_class    {idx: class_name}，正好是前者的反轉；idx 連續覆蓋
                      0 .. len(CLASS_NAMES)-1，因為它由 CLASS_NAMES 推導而
                      不是從 plan 觀察——即使某個 split 缺了一個類別也不會
                      出現斷號（infer_single.py 是用 range 逐格讀的）。
    後兩者是模組級 CLASS_TO_IDX / IDX_TO_CLASS 的複本，呼叫端改動它們不會
    汙染模組狀態，也不會影響已經建好的 dataset。
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
