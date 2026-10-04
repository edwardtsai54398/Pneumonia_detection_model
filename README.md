# Chest X-ray 肺炎分類（EfficientNet_B3 遷移學習 / PyTorch）

以 Kaggle 的 Chest X-Ray Pneumonia 資料集（5856 張胸腔 X 光）對 `efficientnet_b3` 做遷移學習，
執行 `NORMAL` / `PNEUMONIA` 二元分類。每次訓練輸出一組權重、訓練紀錄與圖表到 `outputs/<run_id>/`，
並可用 `infer_single.py` 對單張影像推論。

資料集：<https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/edwardtsai54398/Pneumonia_detection_model/blob/master/notebooks/efficient_train_colab.ipynb)

## 結果

目前 repo 裡還沒有已提交的 `outputs/`，因此沒有可引用的實測數字。

| 指標 | Test |
| --- | --- |
| accuracy | TODO |
| precision | TODO |
| recall | TODO |
| f1 | TODO |

切分方式：`split_scheme = regrouped_70_15_15`、`split_seed = 42`。
跑完一次訓練後，數字請從 `outputs/<run_id>/metadata.json` 的 `test_metrics` 填入。

## Quick start

兩條路徑選一條。Colab 有免費 GPU，是主要路徑。

### A. Colab（推薦）

1. 點上方 **Open In Colab** badge。
2. 切換到 **執行階段 → 變更執行階段類型 → GPU**。
3. 準備 Kaggle 憑證：Kaggle → 右上頭像 → **Settings → API → Create New Token**，
   下載 `kaggle.json`。在 Colab 左側點 **Secrets**，新增兩筆並都開啟 **Notebook access**：

   | Name | Value |
   | --- | --- |
   | `KAGGLE_USERNAME` | `kaggle.json` 的 `username` |
   | `KAGGLE_KEY` | `kaggle.json` 的 `key` |

4. 由 Cell 1 開始依序執行到 Cell 11。Cell 2 會要求授權掛載 Google Drive——
   權重寫到 Drive 才能在執行階段結束後保留下來，`outputs/` 的文字產物仍寫在 clone 出來的 repo 裡。
5. 要把這次的訓練紀錄推回 GitHub，再執行 Cell 12（需要另一個 secret `GH_TOKEN`，
   見下方「改程式碼的流程」）。

### B. 本機

需要 Python 3.10 以上。有 CUDA GPU 會快得多，純 CPU 也能執行。
在專案根目錄執行（PowerShell）：

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Kaggle 憑證放在 `~/.kaggle/kaggle.json`，或設定 `KAGGLE_USERNAME` / `KAGGLE_KEY` 環境變數。
接著開啟 `notebooks/efficient_train.ipynb`，由 Cell 1 依序執行到 Cell 10。

## 專案結構

```text
.
├── constant.py                      # 共用常數：ImageNet mean/std、模型對照表、CLASS_NAMES、DEFAULT_SPLIT_SEED
├── env.py                           # Colab 偵測、collect_run_info()（git commit 與環境版本快照）
├── utils.py                         # set_seed、EpochTimer、collect_config、make_run_id/make_output_dir、save_results
├── infer_single.py                  # 單張影像推論的 CLI
├── requirements.txt                 # 套件版本下限
├── data/
│   ├── dataset.py                   # Kaggle 憑證與下載、transform、ImageFolder、DataLoader
│   └── split.py                     # 索引式 70/15/15 重新切分、病人分組、SampleListDataset
├── engine/
│   ├── trainer.py                   # 單一 epoch 迴圈、train_model（backbone 解凍、early stopping）、指標計算
│   └── visualize.py                 # plot_history、plot_cm、plot_class_distribution
├── models/
│   ├── builder.py                   # build_model（efficientnet_b3 / resnet18 / resnet50，可凍結 backbone）
│   └── inference.py                 # predict（softmax + argmax 批次推論）
├── notebooks/
│   ├── efficient_train.ipynb        # 本機訓練（10 個 code cell）
│   └── efficient_train_colab.ipynb  # Colab 訓練（13 個 code cell，多了 clone / Drive / push）
└── outputs/<run_id>/                # 每次訓練的產物（執行過才會出現）
```

所有訓練邏輯都在根目錄的 `.py` 模組裡，notebook 只負責設定參數與依序呼叫。
要改行為請改 `.py`，不要把函式貼回 notebook cell。

## 資料集

`data/dataset.py` 的 `download_dataset()` 透過 `kagglehub` 下載
`paultimothymooney/chest-xray-pneumonia`，再由 `find_split_root()` 自動找出真正含
`train/val/test` 的那一層（資料集不同版本會多包一層 `chest_xray/`）。

已經有本機副本的話，把設定 cell 的 `DATASET_LOCAL_DIR` 指向下列結構的最上層目錄，就會直接沿用：

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

### 兩種切分方式

| 切分 | 內容 |
| --- | --- |
| 原始 Kaggle 切分 | 資料集附帶的 `train/val/test`。`val` 只有 16 張，用它做 early stopping 與選模型等於在選噪音。notebook 只以 `count_images()` 清點並繪圖，不拿它訓練 |
| `regrouped_70_15_15_grouped_by_patient` | `collect_all_images()` 收齊全部 5856 張，再由 `plan_splits()` 按類別分層切成 70/15/15，且同一位病人的影像整組落在同一個 split。這是實際訓練用的切分，也是寫進 `metadata.json` 的 `split_scheme` 值 |

不同 `split_scheme` 的數字不要直接比較，測試集的組成不同。

### 病人層級的分組

`patient_group_key()` 從檔名取出病人編號（`person1_bacteria_1.jpeg` → `person1`），
`plan_splits(group_key=...)` 再讓同一位病人的所有影像進同一個 split。沒有這層分組時，
同一位病人的 A 片在 train、B 片在 test，模型可以靠「認得這個人」而不是「認得肺炎」
拿到高分，測試指標會偏樂觀。設定 cell 的 `GROUP_BY_PATIENT = False` 可退回逐張切分。

分組後的切分單位是不可分割的群組，所以兩種模式用兩種分配法：逐張切分用最大餘額法
（比例精確），分組切分用貪婪裝箱（大組先放，每次放進張數離配額最遠的 split）。後者
讓張數盡量接近 70/15/15 但不保證精確——誤差下限由群組大小分佈決定。切分 cell 會印出
每個類別的群組數，若群組數等於張數，表示檔名沒有對上病人編號、分組等於沒開。

## 訓練

改參數只改設定 cell：Colab 版是 Cell 3，本機版是 Cell 2。

| 參數 | 說明 |
| --- | --- |
| `EXPERIMENT_NAME` | 實驗名稱，會成為 `run_id` 的前綴 |
| `MODEL_NAME` | `efficientnet_b3` / `resnet18` / `resnet50`，取自 `constant.MODELS` |
| `IMAGE_SIZE` | 輸入解析度，預設 224 |
| `BATCH_SIZE` / `EPOCHS` / `LR` / `WEIGHT_DECAY` | 批次大小、最大 epoch 數、學習率、weight decay |
| `RANDOM_SEED` | 只影響權重初始化、augmentation 與 batch 順序 |
| `SPLIT_SEED` | 只影響 train/val/test 怎麼切。比較不同 `RANDOM_SEED` 時必須固定 |
| `TRAIN_RATIO` / `VAL_RATIO` / `TEST_RATIO` | 切分比例，三者必須加總為 1 |
| `GROUP_BY_PATIENT` | 同一位病人的影像是否整組進同一個 split，預設 `True`。關掉會退回逐張切分 |
| `FREEZE_BACKBONE` | 前幾個 epoch 是否只訓練分類頭 |
| `UNFREEZE_EPOCH` | 在第幾個 epoch 解凍 backbone。`None` 表示全程不解凍 |
| `BACKBONE_LR_FACTOR` | 解凍後 backbone 的學習率 = `LR * BACKBONE_LR_FACTOR` |
| `PATIENCE` | early stopping 的耐心值，監看驗證集 F1 |
| `NUM_WORKERS` | DataLoader 工作進程數。Windows 上建議設為 `0` |
| `DATASET_LOCAL_DIR` | 指向本機既有資料集，`None` 表示從 Kaggle 下載 |

設定 cell 最後的 `CONFIG = collect_config(globals())` 會把上述參數收進 `metadata.json`。
變數名打錯或漏掉會在執行該 cell 時立即 `KeyError`，不會等到訓練結束才發現。

### Cell 順序

| Colab | 本機 | 做什麼 |
| --- | --- | --- |
| 1 | — | clone 或 pull repo、`pip install -r requirements.txt` |
| 2 | 1 | 匯入模組、偵測裝置、掛載 Drive（本機版沒有 Drive） |
| 3 | 2 | **實驗設定** |
| 4 | 3 | 下載資料與清點 |
| 5 | 4 | 原始切分的類別分佈圖 |
| 6 | 5 | 重新切分 70/15/15（按病人分組），印出群組數與各 split 的類別分佈 |
| 7 | 6 | `set_seed()`、建立 dataset / dataloader / model / optimizer / scheduler |
| 8 | 7 | 訓練 |
| 9 | 8 | 測試集評估 |
| 10 | 9 | 訓練曲線與混淆矩陣 |
| 11 | 10 | 儲存結果（Colab 版另外產生 `pip_freeze.txt`） |
| 12 | — | 把 `outputs/` commit 並 push 回 GitHub（選配） |
| 13 | — | `git pull --ff-only` 同步最新程式碼 |

### 重跑某個 cell 的注意事項

- 改了設定 cell 的參數，就從設定 cell 往後依序重跑到儲存結果那個 cell。
- `set_seed()` 在準備模型的 cell（Colab Cell 7 / 本機 Cell 6），不在設定 cell。
  重跑訓練前請連這個 cell 一起重跑，否則 RNG 狀態已經往前推進，權重初始化不會與上次相同。
- 換過執行階段之後，舊的 `plan` 裡的路徑已經失效。請重跑下載與切分的 cell；
  `build_datasets_from_plan()` 會在訓練開始前檢查所有路徑並直接報錯。

## 輸出產物

每次訓練建立一個 `outputs/<run_id>/`，`run_id` 的格式是 `<EXPERIMENT_NAME>_<YYYYmmdd_HHMMSS>`。

| 檔案 | 進版控 | 說明 |
| --- | --- | --- |
| `metadata.json` | 是 | 超參數、環境與程式碼版本、切分資訊、測試集指標 |
| `history.json` | 是 | 每個 epoch 的 train/val loss、f1、accuracy、precision、recall 與 lr |
| `training_curves.png` | 是 | 訓練曲線，2×3 共六張子圖 |
| `confusion_matrix.png` | 是 | 測試集的列正規化混淆矩陣 |
| `pip_freeze.txt` | 是 | 當次環境的精確套件版本，要完整重現環境時照它安裝。僅 Colab Cell 11 產生 |
| `best_model.pth` | 否 | 權重約 47MB，由 `.gitignore` 排除。Colab 執行時寫到 Drive 的 `Pneumonia_Project/weights/<run_id>/` |

### `metadata.json` 的重要欄位

| 欄位 | 說明 |
| --- | --- |
| `model_name` | 模型架構。`infer_single.py` 靠這個決定要建立哪一個 backbone |
| `img_size` | 訓練時的輸入解析度，推論會沿用同一個值 |
| `class_to_idx` / `idx_to_class` | 類別與索引的對應。`infer_single.py` 用它把預測索引轉成標籤 |
| `best_val_f1` | 驗證集最佳 F1，也是選取 checkpoint 的依據 |
| `best_epoch` | 被存下來的那一個 epoch（0-based，對得上 `history.json` 的 `epoch`）。配合 `history` 的長度就看得出 early stopping 是在最佳點之後多跑了幾個 epoch |
| `test_metrics` | 測試集的 loss / accuracy / precision / recall / f1 與 tp、tn、fp、fn |
| `num_parameters` | 模型參數量，單位為百萬 |
| `train_time_per_epoch` | 每個 epoch 的平均秒數 |
| `config` | 設定 cell 的完整參數快照 |
| `git_commit` | 執行時 `HEAD` 的完整 SHA |
| `python_version` / `torch_version` / `torchvision_version` / `cuda_version` / `gpu` / `platform` / `in_colab` | 環境快照 |
| `deterministic` | 固定為 `false`，表示同一個 seed 重跑不保證 bit-level 一致 |
| `split_scheme` / `split_seed` / `split_ratios` / `split_counts` | 切分方式、種子、比例與各 split 的類別分佈 |
| `dataset_id` / `dataset_version` | Kaggle 資料集 ID 與版本號（共用快取路徑解析不到版本時為 `null`） |

## 單張推論

在專案根目錄執行（PowerShell）。三個參數都是必填：

```powershell
python infer_single.py `
  --image_path "C:\path\to\your\xray_image.jpeg" `
  --model_path "outputs\<run_id>\best_model.pth" `
  --metadata_path "outputs\<run_id>\metadata.json"
```

權重與 `metadata.json` 必須來自同一次訓練。`model_path` 指向從 Drive 下載回來的
`best_model.pth` 也可以，`metadata_path` 則指向對應 `run_id` 的 `metadata.json`。
輸出為預測標籤、信心值，以及每個類別的機率。

## 重現某次實驗

打開那次實驗的 `outputs/<run_id>/metadata.json`，依序執行：

1. 把 `git_commit` 的值填進 Colab Cell 1 的 `PIN_COMMIT`，執行該 cell 會 checkout 到那一版程式碼。
   本機則自行執行 `git checkout <SHA>`。
2. 照 `config` 欄位把設定 cell 的每個參數填回去，`RANDOM_SEED` 與 `SPLIT_SEED` 都要對上。
3. 確認 `split_scheme` 與 `dataset_id` 一致。`dataset_version` 不同的話，資料本身就可能已經變動。
4. 執行到切分 cell（Colab Cell 6 / 本機 Cell 5），比對印出的各 split 類別張數與 metadata 裡的
   `split_counts`。**不同就代表兩次實驗的測試集不一樣，指標不能互相比較**，請回頭檢查
   `SPLIT_SEED`、三個比例與 `GROUP_BY_PATIENT`。注意張數相同不保證名單相同——切分程式本身改過的話，
   同一個種子也可能切出不同的名單，所以第 1 步的 `git_commit` 要先對上。
5. 需要完全一致的環境時，依 `pip_freeze.txt` 安裝。在 Colab 上不建議強制 pin `torch`，
   會觸發數 GB 下載，且可能與 driver 不匹配。
6. 執行後續 cell 完成訓練，再比對 `test_metrics`。

## 改程式碼的流程

標準路徑：**在 VS Code 編輯 → commit → push → 回 Colab 執行 Cell 13（`git pull --ff-only`）**。
有 `%autoreload 2`，不需要重啟執行階段，`model` / `history` / `best_state` 都還在。

Cell 12 需要 GitHub token：GitHub → Settings → Developer settings →
**Fine-grained personal access token**，只勾這一個 repo、權限給 **Contents: Read and write**，
存進 Colab Secrets 的 `GH_TOKEN`。

### 要改程式碼前請注意

- **訓練進行中不要執行 Cell 13 或任何 `git pull`。** epoch 是在呼叫時才解析函式，
  中途換掉程式碼會讓 `metadata.json` 記下的 `git_commit` 不等於實際執行的程式碼，而且不會有任何警告。
- **訓練前先 commit 並 push。** `git_commit` 只記錄 `HEAD` 的 SHA，沒有程式檢查工作區是否乾淨、
  或該 commit 是否已經推上遠端。未 commit 的修改、以及只存在 Colab VM 的 commit，都會讓這個欄位失去意義。
- 不要調整 `constant.CLASS_NAMES` 的排序。`ImageFolder` 按字母排序決定索引
  （`NORMAL=0, PNEUMONIA=1`），`data/split.py` 從同一份清單推導對應；兩者不一致會讓舊的
  `best_model.pth` 配新的 `metadata.json` 反向預測，而且不會拋出例外。
- 不要移除或改名 `metadata.json` 的 `model_name` / `img_size` / `idx_to_class`。
  `infer_single.py` 讀這三個欄位，改掉會讓既有的 checkpoint 無法推論。新增欄位是安全的。
- 往設定 cell 新增參數時，同步加進 `utils.CONFIG_KEYS`，否則該參數不會被記錄進 `metadata.json`。

## 疑難排解

| 問題 | 原因 | 解法 |
| --- | --- | --- |
| 想比較不同 seed 的結果 | — | 只改 `RANDOM_SEED`，`SPLIT_SEED` 保持不動，資料切分不會跟著變 |
| 改了 `constant.py` / `utils.py` 的常數卻沒有生效 | `%autoreload 2` 只替換函式本體，模組層級的常數不會更新 | 重啟執行階段，從 Cell 1 重跑 |
| 訓練途中 `FileNotFoundError`，訊息提到資料集快取可能已失效 | 換過執行階段，`plan` 裡的路徑指向已消失的快取 | 重跑下載與切分的 cell |
| `KeyError: 設定 cell 缺少這些變數` | 設定 cell 的變數名打錯或漏掉 | 照訊息列出的名稱修正設定 cell |
| `FileNotFoundError: 在 ... 底下找不到同時含 train/val/test 的資料夾` | `DATASET_LOCAL_DIR` 指到錯誤的層級 | 指向含 `train/val/test` 的那一層，或改回 `None` 從 Kaggle 下載 |
| 下載資料時跳出登入輸入框 | 讀不到 Kaggle 憑證 | 檢查 Colab Secrets 兩筆名稱是否正確且已開啟 Notebook access；本機檢查 `~/.kaggle/kaggle.json` |
| Windows 上 DataLoader 停住或崩潰 | notebook 中的多進程 DataLoader 在 Windows 容易出問題 | 設定 `NUM_WORKERS = 0` |
| Colab 執行階段結束後找不到權重 | 權重寫在 VM 的本機磁碟 | 確認 Cell 2 的 Drive 掛載成功，`WEIGHTS_BASE` 才會指向 Drive |
| `ValueError: train/val/test 比例必須加總為 1` | 三個 ratio 加總不等於 1 | 修正 `TRAIN_RATIO` / `VAL_RATIO` / `TEST_RATIO` |

## 已知限制

- `set_seed()` 沒有設定 `cudnn.deterministic`，`DataLoader` 也沒有固定 worker 種子，
  同一個 seed 重跑不保證 bit-level 一致。固定 `SPLIT_SEED` 消除的是「測試集不同」這個混淆因子，
  不代表小幅度的指標差異有意義。
- 沒有 CLI 訓練腳本，訓練流程只能由 notebook 執行。
- 決策方式固定為 argmax，沒有針對 recall 偏好調整分類門檻。
- `requirements.txt` 只記錄版本下限，實際環境以每次訓練的 `pip_freeze.txt` 為準。
- 本專案為學習用途，不是醫療器材，不可用於臨床診斷。
