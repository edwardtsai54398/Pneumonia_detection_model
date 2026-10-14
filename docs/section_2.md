# Section 2｜資料前處理與探索性分析 — 程式工作清單

> 對應 [agenda.md](agenda.md) 第二章。
>
> **狀態更新**：本章原本最大的缺口（自訂 stratified 切分完全沒實作）**已經做完，而且做得比原本規劃的更完整**——
> [data/split.py](../data/split.py) 以索引式切分取代了「建實體資料夾／symlink」的做法，原規劃的五個實作陷阱全部處理掉了。
>
> 剩下的缺口集中在**探索性分析（EDA）**：影像屬性掃描、mean/std 對照、重複影像檢查、augmentation 可視化全都還沒有。
> 也就是說，第二章現在的狀況反過來了——「前處理」有紮實的程式支撐，「分析」還是空的。

## 一、現有程式碼已覆蓋

### 前處理

| 需求 | 現況 | 位置 |
| --- | --- | --- |
| 色彩模式標準化 | `Grayscale(num_output_channels=3)` 把 `L` 與 `RGB` 統一成預訓練骨幹要的三通道 | [data/dataset.py:101](../data/dataset.py#L101) |
| 尺寸標準化 | train 走 `Resize((256,256)) → RandomResizedCrop(224, scale=0.85~1.0)`；eval 走 `Resize((224,224))` | [data/dataset.py:98-120](../data/dataset.py#L98-L120) |
| 數值標準化 | ImageNet mean/std | [constant.py:1-2](../constant.py#L1-L2) |
| Augmentation | 隨機縮放裁切、±7° 旋轉、水平翻轉 | [data/dataset.py:103-105](../data/dataset.py#L103-L105) |
| 類別不平衡處理 | `compute_class_weights()` 算出反頻率權重餵給 `CrossEntropyLoss` | [engine/trainer.py:6-11](../engine/trainer.py#L6-L11) |

### 切分（原 2.6，已完成）

| 需求 | 現況 | 位置 |
| --- | --- | --- |
| 分層切分 | `plan_splits()` 逐類別各自切分；比例分配用最大餘額法，保證三份加總等於總數（不會像 `int()` 那樣掉一張） | [data/split.py:106-166](../data/split.py#L106-L166)、[data/split.py:88-103](../data/split.py#L88-L103) |
| 不建實體資料夾 | 切分只產生 `{split: [(path, class_idx)]}` 名單，不建 symlink 樹——檔名碰撞、唯讀快取失效、Windows 權限這三類問題一次消失 | [data/split.py:1-10](../data/split.py#L1-L10) |
| train/val 套不同 transform | `SampleListDataset` 為每個 split 建**獨立實例**，各自帶自己的 transform，val 不會被餵進隨機裁切與翻轉 | [data/split.py:203-233](../data/split.py#L203-L233)、[data/split.py:257-275](../data/split.py#L257-L275) |
| 官方 16 張 val 的處置 | 已決定：全部收回資料池重切，`split_scheme = regrouped_70_15_15`。報告裡不會有兩個都叫 val 的東西 | [data/split.py:52-85](../data/split.py#L52-L85) |
| 切分結果寫進 metadata | `split_scheme` / `split_seed` / `split_ratios` / `split_counts` / `dataset_id` / `dataset_version` | notebook 切分 cell 的 `split_info`、[utils.py:153-158](../utils.py#L153-L158) |
| 類別索引來源 | `CLASS_TO_IDX` 從 `CLASS_NAMES` 推導，不從觀察到的資料推導（某個 split 缺類別時不會產生不連續索引） | [data/split.py:28-29](../data/split.py#L28-L29) |
| 收集階段的防呆 | 遇到 `CLASS_NAMES` 以外的目錄名直接 raise，不做「非 NORMAL 就當 PNEUMONIA」的歸類；全域排序讓結果不依賴走訪順序 | [data/split.py:60-85](../data/split.py#L60-L85) |
| 路徑有效性檢查 | `_verify_paths()` 訓練前檢查**全部**路徑（刻意不抽樣），快取失效會在開始前大聲失敗，而不是第三個 epoch 才在 PIL 裡爆 | [data/split.py:236-254](../data/split.py#L236-L254) |
| 切分種子與訓練種子分離 | `SPLIT_SEED` ≠ `RANDOM_SEED`；參數名刻意叫 `split_seed`，誤傳 `seed=` 會直接 `TypeError` | [constant.py:21-24](../constant.py#L21-L24)、[data/split.py:106-125](../data/split.py#L106-L125) |
| 病人分組切分的介面 | `plan_splits(group_key=...)` 可改為整組進同一個 split，預設 `None`（關閉） | [data/split.py:113](../data/split.py#L113)、[data/split.py:142-149](../data/split.py#L142-L149) |

## 二、缺口

1. **沒有任何 EDA。** 「影像尺寸與色彩模式的混雜狀況」目前是常識性推測，沒有掃描過這份資料集。報告需要實際分布才站得住腳。
2. **標準化策略沒有對照依據。** 用 ImageNet mean/std 是合理選擇，但沒有算過本資料集自己的 mean/std，等於沒有「策略」只有「預設值」。
3. **類別不平衡只做了處理、沒有做分析。** `compute_class_weights()` 的輸出直接進 loss，數值從未被記錄進 `metadata.json`（`CONFIG_KEYS` 裡沒有它，[utils.py:44-62](../utils.py#L44-L62)），也沒有和其他方案比較過，無法支撐 agenda 寫的「決定處理方式」。
4. **切分名單沒有逐檔存檔。** metadata 只記了 `split_seed` / `split_ratios` / `split_counts`，沒有任何東西能讓第二組實驗讀回第一組的名單——而 [section_3.md](section_3.md) 的 3.2 對照實驗要求「切分在第一組產生後固定下來，之後每組都讀同一份」。目前每組實驗都是各自重跑 `plan_splits()`，只靠 seed 隱性保證一致。而 seed 並不足以保證：ratios、`GROUP_BY_PATIENT`、輸入檔案集合、甚至切分程式本身改過，都會讓同一個種子切出不同的名單。
5. **沒有斷言檢查。** `plan_splits()` 驗證了比例合法性（[data/split.py:127-133](../data/split.py#L127-L133)），但沒有驗證結果：三個 split 的正類別比例差距、以及兩兩檔案集合的交集必須為空。由建構方式看這兩件事成立，但沒有程式把它釘住——改動切分邏輯時不會有東西攔你。
6. **沒有跨 split 的重複影像檢查，而重切讓這件事變得更重要。** 原本「官方 train 與官方 test 是否有重複」是資料集本身的問題；現在三個 split 被打散重組，同一張圖的兩個副本落在新 train 與新 test 的機率更高，而這會直接汙染第五章的測試集數字。
7. **病人層級的洩漏沒有被量化。** `group_key` 的介面有了但預設關閉，README 已列為已知限制，可是沒有任何數字說明影響多大。
8. **Augmentation 效果沒有可視化**，報告無法說明「做了哪些增強、強度合不合理」。
9. **隨機性控制仍不完整。** `set_seed()` 只設了 `random` / `numpy` / `torch` / `cuda`（[utils.py:16-21](../utils.py#L16-L21)），缺 `cudnn.deterministic` 與 DataLoader 的 worker 種子。這件事目前的處置是**誠實記錄而非修掉**——`collect_run_info()` 固定寫 `"deterministic": false`（[env.py:68-71](../env.py#L68-L71)）。記錄比假裝好，但第五章要比較兩種訓練策略，前提是除了策略以外其他都一樣。
10. **沒有 sampler 選項。** `make_dataloaders()` 仍是固定 `shuffle=(split == "train")`（[data/dataset.py:146-156](../data/dataset.py#L146-L156)），沒有 `WeightedRandomSampler` 這個對照方案。

## 三、要做的程式工作

### 2.1 影像屬性掃描（EDA 主體）

- 位置：新檔 `data/eda.py`
- 做什麼：走訪全部影像，**只讀檔頭不解碼像素**（PIL 開檔後直接取 `size` 與 `mode`，不要呼叫 `load()` 或轉 tensor，否則 5,856 張會跑很久也吃記憶體）。每張收集：相對路徑、原始 split、類別、寬、高、長寬比、色彩模式、檔案大小。
- 輸入來源：直接吃 `collect_all_images()` 的結果，不要自己再寫一次目錄走訪——那會變成第二套「什麼算影像檔」的定義。
- 統計輸出：
  - 寬／高的 min、max、中位數、5% 與 95% 分位數
  - 色彩模式的次數分布（預期混有 `L` 與 `RGB`，可能還有少量其他）
  - 長寬比分布，以及偏離 1.0 的程度
- 產出：逐張明細存 `docs/assets/image_meta.csv`（2.4 洩漏檢查、第五章挑誤判樣本都會用到），摘要存 `docs/assets/eda_summary.json`。
- 報告怎麼用：
  - 色彩模式混雜 → 證明 `Grayscale(num_output_channels=3)` 是必要的（不是為了「轉灰階」，而是為了把 `L` 與 `RGB` 統一成三通道）。
  - 尺寸落差大且長寬比不一 → 說明為何用固定 `Resize` 而非保持比例填黑邊，以及這個選擇的形變代價。

### 2.2 尺寸與長寬比分布圖

- 位置：[engine/visualize.py](../engine/visualize.py)
- 做什麼：寬高散佈圖（點標類別）＋ 長寬比直方圖，圖上標出 224 與 256 的參考線，讓「縮放到 224 要丟掉多少資訊」看得出來。**軸標籤與標題用英文。**
- 要一起講清楚的一件事：train 的形變發生**兩次**——先 `Resize((256,256))` 把任意長寬比強制壓成正方形，再 `RandomResizedCrop` 以預設 `ratio=(3/4, 4/3)` 又加一層長寬比抖動。eval 只壓一次。所以 train 與 eval 的幾何分布本來就不同，這點值得在報告裡說明是有意識的取捨（augmentation 的一部分），而不是看漏了。
- 產出：`docs/assets/fig_2_size_dist.png`。

### 2.3 標準化策略的對照數據

- 位置：`data/eda.py`
- 做什麼：從**重切後的 train split 抽樣**（固定 seed 抽 500–1000 張，不需全掃）計算本資料集的灰階 mean/std，與 ImageNet 的 `[0.485,0.456,0.406] / [0.229,0.224,0.225]` 並列。
- 注意：要從 `plan["train"]` 抽，不要從官方 train 目錄抽。否則算出來的統計量與實際餵進模型的那批資料不是同一群。
- 報告怎麼用：寫出兩組數字的差距，然後說明**仍採用 ImageNet 常數**的理由——預訓練權重的第一層卷積是在該輸入分布下學到的，換成資料集自身統計量會讓輸入分布偏離預訓練假設，遷移學習的前幾層反而要重新適應。這是有對照數字的決策，不是沿用預設值。
- 產出：併入 `eda_summary.json`。

### 2.4 跨 split 重複影像與病人洩漏檢查

- 位置：`data/eda.py`
- 做什麼：
  1. 對每個檔案算內容 hash（SHA-1，讀 bytes 即可），找出 hash 相同但落在**重切後**不同 split 的影像。
  2. 從檔名解析病人識別碼（PNEUMONIA 側的檔名形如 `personNNN_virus_M.jpeg`，同一個 `personNNN` 就是同一位病人），統計有多少位病人的影像同時出現在 train 與 test、涉及幾張圖。
- 為什麼要兩件一起做：hash 查的是「同一張圖的副本」，檔名查的是「同一位病人的不同張」。後者才是這份資料集真正的洩漏來源，而且目前預設是開著的（`group_key=None`）。README 把它列為已知限制，但「偏樂觀」需要一個數字才有意義——跑出來可能是 30% 的測試張數都有同病人出現在 train，也可能只有 3%，兩者對結論強度的影響完全不同。
- 連帶的決策：量化之後再決定要不要把 `group_key` 打開。打開會讓實際張數不再精確等於 70/15/15（比例是套在病人數上），這個取捨要寫進報告。
- 產出：`docs/assets/duplicate_report.csv`（可能為空）、`docs/assets/patient_leakage.json`。**查出來是零也要寫進報告**——這是一句很便宜的可信度加分。

### 2.5 類別不平衡量化與方案比較

- 位置：`data/eda.py` ＋ [engine/trainer.py](../engine/trainer.py) ＋ [data/dataset.py](../data/dataset.py)
- 做什麼：
  1. 算出重切後 train/val/test 各自的 NORMAL:PNEUMONIA 比例。因為是分層切分，三者應該幾乎一致（約 1:2.7，由全體 1583:4273 決定）——**這本身就是重切的成果之一**，與第一章記錄的官方比例（train 1:2.89、test 1:1.67）對照著寫。
  2. 把 `compute_class_weights()` 的輸出值寫進 `metadata.json`（見 [section_3.md](section_3.md) 的 3.4）——報告附錄要寫「NORMAL 權重 x、PNEUMONIA 權重 y」，必須來自程式。注意它不在 `CONFIG_KEYS` 裡，因為它是算出來的而不是設定的，要走 `save_results()` 的另一條路加進去。
  3. 讓 `make_dataloaders()` 多一個可選的取樣器開關，支援 `WeightedRandomSampler`，使「class weight」與「over-sampling」成為可切換的兩個選項。
- 為什麼要加 sampler 選項：agenda 寫的是「決定處理方式」。要能說「決定」，就得有第二個選項可比。只做其中一個就只能說「採用」。實作上注意 sampler 與 `shuffle=True` 互斥，只能二選一。
- 驗收：開啟 sampler 後，統計一個 epoch 實際取到的標籤分布應接近 1:1。

### 2.6 切分名單存檔與斷言（補完已完成的切分）

- 位置：[data/split.py](../data/split.py)
- 做什麼：
  1. **匯出逐檔名單**：輸出「`split/class/filename` → 新 split」的 CSV 或 JSON 到 `outputs/<run_id>/`。識別鍵用原始樹裡的 `split/class/filename` 相對路徑而非絕對路徑——Colab 的根目錄是 `/kaggle/input/...`、本機是別的位置，用絕對路徑的話同一份切分在兩台機器上會對不起來。
  2. **提供讀回名單的函式**，讓 `build_datasets_from_plan()` 可以吃一份既有名單而不是現場重切。這是 [section_3.md](section_3.md) 3.2 對照實驗的前提：讀同一份名單才能從一開始就保證每組實驗的 test 完全一樣，而不是事後才發現切分錯開。
  3. **加斷言**：三個 split 的正類別比例差距小於容忍值；兩兩檔案集合交集為空。失敗就直接停。
- 為什麼 1、2 是必要的：對照實驗跑了四十分鐘才發現切分錯開，成本太高。名單本身就是切分的憑據。**已決定不走「存一個切分摘要／雜湊值」那條路**：那種值只能回答「兩次一不一樣」，不能回答「差在哪」，也不能讓第二組實驗讀回同一份切分，等於用維護成本換一個不夠用的答案。
- 驗收：第二組實驗讀第一組匯出的名單，`split_counts` 與逐檔名單都要逐一相符；刻意把某個類別的檔案複製到兩個 split，交集斷言要攔下來。

### 2.7 Augmentation 效果可視化

- 位置：[engine/visualize.py](../engine/visualize.py)
- 做什麼：挑一張 train 影像，重複套 train transform 6–8 次，連同原圖排成網格（要**反標準化**才看得出來，否則畫面會是過曝的雜訊）。
- 報告怎麼用：圖 2-3 搭配說明為何旋轉只給 ±7°、為何保留水平翻轉——胸腔 X 光的拍攝姿位固定，過大的旋轉與垂直翻轉會產生臨床上不存在的影像，反而是引入錯誤先驗。同時讓 2.2 提到的「兩次形變」看得見。
- 產出：`docs/assets/fig_2_augmentation.png`。

### 2.8 隨機性控制補強

- 位置：[utils.py:16-21](../utils.py#L16-L21) 與 [data/dataset.py:146-156](../data/dataset.py#L146-L156)
- 做什麼：`set_seed()` 還缺——
  - DataLoader 的 `generator` 與 `worker_init_fn`（`num_workers > 0` 時各 worker 的隨機狀態目前不受控，augmentation 序列無法重現。Colab 設定是 `NUM_WORKERS = 2`，所以這在主要執行環境上是實際發生的）
  - `cudnn.deterministic` / `cudnn.benchmark` 的取捨（決定論換速度，建議開啟決定論）
- 連帶要做：改完之後把 [env.py:71](../env.py#L71) 寫死的 `"deterministic": False` 改成反映實際設定值。現在這個欄位是誠實的，改了程式卻沒改它就會變成謊話。
- 為什麼放在本章：切分與 augmentation 都靠隨機數。切分那半有 `SPLIT_SEED` 管著（真正鎖死要等 2.6 的名單存檔），augmentation 這半連種子都還不受控。第五章要比較兩種訓練策略，**除了策略以外其他都一樣**是前提；這件事不補，整章對照實驗的結論強度都會被打折。

## 四、本章報告素材清單

| 素材 | 來源任務 | 檔案 |
| --- | --- | --- |
| 尺寸／色彩模式統計摘要 | 2.1 | `eda_summary.json` |
| 逐張影像屬性明細 | 2.1 | `image_meta.csv` |
| 圖 2-1 尺寸與長寬比分布 | 2.2 | `fig_2_size_dist.png` |
| 本資料集 vs ImageNet mean/std 對照 | 2.3 | `eda_summary.json` |
| 重複影像檢查結果 | 2.4 | `duplicate_report.csv` |
| 病人層級洩漏的量化 | 2.4 | `patient_leakage.json` |
| 重切前後的類別比例對照 | 2.5 | `eda_summary.json`、第一章的 `dataset_stats.json` |
| class weight 數值 | 2.5 | `metadata.json` |
| 圖 2-2 重切後類別分布 | [section_1.md](section_1.md) 的 1.4 | `fig_1_regrouped_split.png` |
| 切分名單與斷言結果 | 2.6 | `outputs/<run_id>/` 下的 split 名單 |
| 圖 2-3 augmentation 效果 | 2.7 | `fig_2_augmentation.png` |
| 切分設定（實驗可比性的憑據） | 已完成 | `metadata.json` 的 `split_scheme` / `split_seed` / `split_ratios` / `split_counts` |
