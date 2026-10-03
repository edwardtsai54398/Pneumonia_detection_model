# Chest X-ray 肺炎分類（EfficientNet_B3 遷移學習 / PyTorch）

此專案使用 Kaggle 的 Chest X-Ray Pneumonia 資料集，透過 EfficientNet_B3 進行遷移學習，完成二元分類：

- `NORMAL`: 正常胸腔 X 光
- `PNEUMONIA`: 肺炎胸腔 X 光

資料集連結：
`https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data`

## 1) 安裝套件

在專案根目錄下執行：

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2) 模組說明

所有程式碼都在這些 `.py` 裡，**notebook 只負責把它們串起來**。不要把函式定義貼進
notebook——那會變成第二份複本，然後兩邊就會各自演化。

- `constant.py`：共用常數（ImageNet mean/std、模型名稱對照表、類別名稱、預設切分種子）。
- `env.py`：Colab 偵測、以及記錄「這次是在什麼環境、哪一版程式碼跑的」（`collect_run_info()`）。
- `config` 沒有獨立檔案：超參數寫在 notebook 的設定 cell，由 `utils.collect_config()` 收進 `metadata.json`。
- `data/dataset.py`：Kaggle 憑證處理與下載、影像 transform、`ImageFolder` 建構、`DataLoader` 建構。
- `data/split.py`：70/15/15 重新切分（索引式，只產生名單不建資料夾）、切分指紋。
- `engine/trainer.py`：單一 epoch 訓練/驗證迴圈、完整訓練迴圈（含 backbone 解凍排程、early stopping）、指標計算。
- `engine/visualize.py`：訓練曲線、混淆矩陣、類別分佈繪圖。
- `models/builder.py`：模型建構（`resnet18`/`resnet50`/`efficientnet_b3`，可切換 backbone 凍結）。
- `models/inference.py`：批次推論（softmax + argmax）。
- `utils.py`：亂數種子、計時器、設定快照、輸出目錄與結果的儲存。
- `notebooks/`：`efficient_train.ipynb`（本機）與 `efficient_train_colab.ipynb`（Colab）。

> ⚠️ `constant.CLASS_NAMES` 必須維持字母排序。`ImageFolder` 是按字母排序類別目錄
> 決定索引（`NORMAL=0, PNEUMONIA=1`），而 `data/split.py` 從 `CLASS_NAMES` 推導同一份
> 對應。兩者不一致時，舊的 `best_model.pth` 配新的 `metadata.json` 會**安靜地反向
> 預測**——不丟例外，信心值看起來還很合理。

## 3) 下載與放置資料集

資料集透過 notebook 內的 `download_dataset()` 以 `kagglehub` 自動下載，也可以手動放置成以下結構（重點是 `train/`, `val/`, `test/`）：

```text
chest_xray/
├── train
│   ├── NORMAL
│   └── PNEUMONIA
├── val
│   ├── NORMAL
│   └── PNEUMONIA
└── test
    ├── NORMAL
    └── PNEUMONIA
```

**注意官方切分不直接拿來用**：原始的 `val/` 只有 16 張（NORMAL 8、PNEUMONIA 8），
用它做 early stopping 和選模型基本上是在選噪音。所以 notebook 會把全部 5856 張
重新切成 70/15/15（每個類別各自切，維持類別比例），val 因此變成約 877 張。

## 4) 開始訓練

兩本 notebook，流程一致：

- `notebooks/efficient_train_colab.ipynb` — Colab（主力）
- `notebooks/efficient_train.ipynb` — 本機

本專案以 notebook 訓練，無 CLI 訓練腳本。Cell 順序：

| Cell | 做什麼 |
| --- | --- |
| 1 | （Colab）clone repo、裝套件 ／（本機）匯入模組 |
| 2 | 匯入模組、偵測裝置、掛載 Drive |
| 3 | **實驗設定（換參數只改這個 cell）** |
| 4 | 下載資料與清點 |
| 5 | 原始切分的類別分佈 |
| 6 | 重新切分 70/15/15 |
| 7 | 準備資料集與模型 |
| 8 | 訓練 |
| 9 | 測試集評估 |
| 10 | 訓練曲線與混淆矩陣 |
| 11 | 儲存結果 |
| 12–13 | （Colab）把紀錄 push 回 GitHub、同步最新程式碼 |

（本機版沒有 Colab 專屬的 1 / 12 / 13，所以編號各往前一格。）

### 設定 cell 的兩個種子

```python
RANDOM_SEED = 43     # 只影響權重初始化 / augmentation / batch 順序
SPLIT_SEED  = 42     # 只影響資料切分
```

**不要混用。** 想比較不同 `RANDOM_SEED` 的穩定性時，`SPLIT_SEED` 必須固定——否則
換 seed 會連 train/val/test 怎麼切都一起換掉。實測 `seed=43` 的 NORMAL 測試集有
172/238 張是 `seed=42` 的訓練資料，那樣兩次實驗根本不能比。

設定 cell 最後的 `CONFIG = collect_config(globals())` 會把所有參數收進
`metadata.json`。打錯變數名（`EPOCHSS = 30`）會在跑那個 cell 當下就 `KeyError`，
不會等到訓練完才發現。往設定 cell 加參數時，記得同步加進 `utils.CONFIG_KEYS`。

### 在 Colab 改程式碼的流程

主要路徑：**在 VS Code 改 → commit → push → 回 Colab 跑最後一個同步 cell
（`git pull --ff-only`）**。有 `%autoreload 2`，不用重啟執行階段，`model` /
`history` / `best_state` 都還在。

這條路徑不是「比較乾淨」而已——它是唯一能讓 `metadata.json` 裡那個 SHA 真的
存在於 GitHub 的做法。

> ⚠️ `%autoreload` 只換掉**函式本體**。改了 `constant.py` / `utils.py` 的常數
> （`CLASS_NAMES`、`CONFIG_KEYS`）必須重啟執行階段。
>
> ⚠️ **訓練進行中不要 `git pull`**。epoch 是在呼叫時才解析函式，中途換掉程式碼
> 會讓 `git_commit` 變成謊言，而且沒有任何機制會偵測到。

逃生口：訓練到一半才發現小 bug、又不想丟掉這次 run，就用 Colab 內建編輯器改檔、
重跑那個 cell，然後從 Cell 12 commit + push，別讓修正困在 VM 裡。

### 輸出

每次實驗一個 `outputs/<experiment_name>_<timestamp>/`：

| 檔案 | 進版控？ | 說明 |
| --- | --- | --- |
| `metadata.json` | ✅ | 超參數、`git_commit`、環境版本、切分資訊、測試集指標 |
| `history.json` | ✅ | 每個 epoch 的訓練/驗證指標 |
| `pip_freeze.txt` | ✅ | 精確的套件版本（`requirements.txt` 只有 `>=` 下限）|
| `training_curves.png`、`confusion_matrix.png` | ✅ | 視覺化圖表 |
| `best_model.pth` | ❌ | 47MB，被 `.gitignore` 擋掉；Colab 會寫到 Drive |

所以 GitHub 上每一次實驗 = **一個 commit SHA + 一個 outputs 資料夾**。

## 5) 單張影像推論

```powershell
python infer_single.py `
  --image_path "C:\path\to\your\xray_image.jpeg" `
  --model_path "outputs\<experiment_name>_<timestamp>\best_model.pth" `
  --metadata_path "outputs\<experiment_name>_<timestamp>\metadata.json"
```

`infer_single.py` 會依 `metadata.json` 裡的 `model_name` 自動選擇正確的模型架構（`resnet18`/`resnet50`/`efficientnet_b3`）。

## 6) 重現某次實驗

打開那次實驗的 `outputs/<run>/metadata.json`：

1. **程式碼版本**：把 `git_commit` 的值填進 Colab notebook Cell 1 的 `PIN_COMMIT`，
   它會 checkout 那一版再跑。
2. **超參數**：`config` 欄位就是當時設定 cell 的完整內容，照著填回去。
3. **資料切分**：確認 `split_scheme` 與 `split_seed` 一致，跑完 Cell 6 之後比對
   印出來的 `split_fingerprint` 跟 metadata 裡的是否相同。**不同就代表這兩次實驗
   的測試集不一樣，數字不能互比。**
4. **環境**：`pip_freeze.txt` 是當時的精確版本。torch 不建議在 Colab 上強制 pin
   （會觸發數 GB 下載並有 CUDA/driver 不匹配風險），所以是用「記錄」而不是「固定」。

### `git_commit` 的可信度

這個欄位的正確性**靠你訓練前先 commit 並 push 的紀律，沒有程式在把關**。兩個會
讓它變成虛構的情況：

- 在 Colab 隨手改了模組但沒 commit → `HEAD` 還指著舊 commit，記下的 SHA 不是實際跑的程式碼
- commit 了但沒 push → 那個 SHA 只存在 Colab VM，執行階段一關就消失

真的被咬到的話，`env.collect_run_info()` 加兩行（`git status --porcelain` 和
`git branch -r --contains HEAD`）就能把這兩種情況變成警告。

## 7) 備註

- 本訓練程式已使用 ImageNet 預訓練權重做遷移學習。
- 由於資料集有類別不平衡，loss 會自動使用 class weights。
- 在 Windows 上若 `num_workers` 過大可能不穩定，建議先使用 `0`。
- `set_seed()` 沒有設定 `cudnn.deterministic`，`DataLoader` 也沒有固定 worker 種子，
  所以同一個 seed 重跑**不保證** bit-level 一致（`metadata.json` 的
  `deterministic: false` 就是在說這件事）。固定 `SPLIT_SEED` 移除的是「測試集根本
  不一樣」這個大混淆因子，但不會讓 0.3% 的 F1 差異變得有意義。
- 目前**沒有**處理同一位病人的影像同時落在 train 和 test 的問題。`plan_splits()`
  留了 `group_key` 參數可以做分組切分，預設關閉。
