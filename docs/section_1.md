# Section 1｜緒論與資料集介紹 — 程式工作清單

> 對應 [agenda.md](agenda.md) 第一章。本章論述成分高，程式的任務是**產出可被引用的數字**，不要讓報告裡出現手打的統計值。
>
> **狀態更新**：下載邏輯已經單一來源化（原 1.1 完成），清點與繪圖也已經是模組函式、兩個 notebook 共用。
> 但本章仍有一個論述層面的新問題：程式現在做的切分比第一章原本要論證的更激進——
> 官方的 `train/val/test` **三個都被打散重切**，不只是重切 val。詳見下方 1.3 與 1.6。

## 一、現有程式碼已覆蓋

| 需求 | 現況 | 位置 |
| --- | --- | --- |
| 資料集取得 | `download_dataset()` 串起 `ensure_kaggle_auth()`（環境變數 → Colab Secrets → `~/.kaggle/kaggle.json` → 互動式登入）與 `find_split_root()`（容忍不同版本多包一層 `chest_xray/`），本機與 Colab 共用同一條路徑 | [data/dataset.py:16-95](../data/dataset.py#L16-L95) |
| 樣本數清點 | `count_images()` 逐 split、逐類別數檔案，不寫死副檔名、會跳過非目錄項目 | [data/split.py:32-49](../data/split.py#L32-L49) |
| 類別分布圖 | `plot_class_distribution()` 接受單層或巢狀 counts，英文軸標、長條標數值、色盲安全配色 | [engine/visualize.py:93-147](../engine/visualize.py#L93-L147) |
| 類別名稱／正類別定義 | `POSITIVE_CLASS_NAME = "PNEUMONIA"`；`CLASS_NAMES` 有字母排序斷言，索引由它推導而非由資料推導 | [constant.py:11-19](../constant.py#L11-L19) |
| 資料集版本記錄 | `dataset_version()` 從 kagglehub 快取路徑解析版本號，與 `dataset_id` 一起寫進 `metadata.json` | [data/dataset.py:66-72](../data/dataset.py#L66-L72)、notebook 切分 cell 的 `split_info` |
| 取得時間記錄 | `run_id` 帶時間戳（`<EXPERIMENT_NAME>_<YYYYmmdd_HHMMSS>`），`collect_run_info()` 另記 git commit 與環境版本 | [utils.py:77-79](../utils.py#L77-L79)、[env.py:58-74](../env.py#L58-L74) |

## 二、缺口

1. **清點數字仍然只存在 stdout。** 重切之後的 `split_counts` 有進 `metadata.json`，但**官方原始切分的張數沒有任何檔案留下**——而第一章的論點（官方 val 只有 16 張）正是建立在那組數字上。重跑一次就沒了，報告無法引用。
2. **「官方 val 僅 16 張」仍然只是現象，沒有量化成論點。** 占全資料集的百分比、單一樣本對驗證指標的解析度，程式都沒有算。
3. **圖只有 `show=True`，沒有存檔。** 兩個 notebook 都是直接畫在 cell 輸出裡（Colab Cell 5／本機 Cell 4），`docs/assets/` 目前根本不存在。報告要用圖只能截圖，解析度與可重現性都不行。
4. **總張數有一個未解釋的差異。** README 寫 5856 張，Kaggle 頁面宣稱 5,863。而且 repo 裡有兩套數法：`count_images()` 數**所有檔案**（[data/split.py:45](../data/split.py#L45)），`collect_all_images()` 只收 `IMG_EXTS` 內的副檔名（[data/split.py:76](../data/split.py#L76)）。兩者的差值就是被排除的非影像檔，目前沒有人對過。
5. **沒有樣本影像網格**，第一章無法讓讀者看到兩類影像的肉眼差異有多不明顯。
6. **第一章的論述結構與程式現狀已經不一致（本章最需要處理的一項）。** 程式做的是 `regrouped_70_15_15`——把官方三個 split 全部收齊再重切。「官方 val 太小」只能解釋為什麼要重做 val，**不能解釋為什麼連官方 test 也打散**。這個決定目前在報告裡沒有理由，而它的代價很實際：重切後的測試集不再是文獻上大家回報數字的那 624 張，本專案的指標無法與任何已發表結果對照。

## 三、要做的程式工作

### 1.1 資料集盤點存檔

- 位置：新檔 `data/inventory.py`（或 `data/split.py` 增一個匯出函式）
- 做什麼：把 `count_images()` 的結果轉成結構化產物——每個 split 的 NORMAL 數、PNEUMONIA 數、小計、正類別占比、該 split 占全資料集百分比，加上全體總數。同時記錄 `dataset_id`、`dataset_version`、kagglehub 快取路徑、掃描時間戳。
- 產出：`docs/assets/dataset_stats.json` 與 `dataset_stats.csv`（CSV 方便直接貼進報告表格）。
- 驗收：三個 split 的小計相加等於總檔案數。
- 為什麼不直接沿用 `metadata.json`：那份記錄的是**重切後**的分布，而第一章要講的是**重切前**。兩者混在一起，報告裡會出現兩個都叫「train」但張數不同的東西。

### 1.2 兩套數法的差異對帳

- 位置：同 1.1
- 做什麼：同時輸出「所有檔案數」（`count_images()` 的邏輯）與「符合 `IMG_EXTS` 的檔案數」（`collect_all_images()` 的邏輯），並列出被排除的檔案清單。
- 為什麼：這是一句很便宜的可信度加分。報告說「本資料集 5856 張」時，要能回答「Kaggle 頁面寫 5,863，差的 7 張去哪了」。查出來差值是零也要寫——那就證明 `IMG_EXTS` 的白名單沒有漏掉任何影像。
- 驗收：兩個數字都印出來，差值有明確的檔案清單可對照。

### 1.3 官方切分體檢

- 位置：同 1.1，獨立一個函式
- 做什麼：由盤點結果直接算出第一章要用的數字——
  - 官方 val 的總張數、每類張數（預期 NORMAL 8 / PNEUMONIA 8）
  - val 占全資料集的百分比（預期不到 0.3%）
  - 以 val 張數推算單一樣本對驗證指標的解析度（16 張時，一張錯判就讓 accuracy 跳動 6.25 個百分點）
  - **官方 test 的張數與類別比例**（624 張，約 234 / 390），以及它與官方 train 的類別比例差距
- 為什麼：前三項是「為何重做 val」的論據，第四項是 1.6 要用的——要論證「連 test 也重切」，就得先有官方 test 自己的數字。
- 產出：併入 `dataset_stats.json`，另外輸出一行可直接引用的摘要字串。

### 1.4 圖檔存檔（不只是畫在 cell 裡）

- 位置：notebook 的分布圖 cell（Colab Cell 5 / 本機 Cell 4），搭配 `plot_class_distribution()` 已回傳 figure 物件的設計
- 做什麼：建立 `docs/assets/`，把原始切分與重切後的兩張分布圖都 `savefig(..., dpi=150)`。繪圖函式本身不用改——它已經回傳 `fig`，呼叫端存檔即可。
- 產出：`docs/assets/fig_1_original_split.png`（第一章圖 1-1）、`docs/assets/fig_1_regrouped_split.png`（第二章用）。
- 注意：兩張圖並列才看得出重切前後的差別，這是第一章收尾、第二章開場的那張圖。`docs/assets/` 要加進版控（圖與 JSON 都不大），報告才不會依賴本機暫存檔。

### 1.5 樣本影像網格

- 位置：[engine/visualize.py](../engine/visualize.py) 新增繪圖函式
- 做什麼：固定 seed 從 NORMAL／PNEUMONIA 各抽 4–6 張原圖（**不套任何 transform**），排成兩列網格，標題標類別。**軸標籤與標題用英文**（原因見 [section_5.md](section_5.md) 的字型注意事項）。
- 為什麼：第一章需要讓讀者看到兩類影像的肉眼差異（肺炎側的浸潤／白影）有多不明顯，鋪陳為何需要 CNN 而非規則式判斷。
- 產出：`docs/assets/fig_1_samples.png`（圖 1-2）。

### 1.6 把「為什麼連官方 test 也重切」變成有數字的論證

- 位置：論述工作為主，程式面是保留對照能力
- 做什麼：
  1. 用 1.3 的數字寫出重切的完整理由。可用的論據：官方 val 16 張無法做模型選擇；官方 train/val/test 的類別比例彼此不一致（train 約 1:2.9、test 約 1:1.7），用一個比例失衡的 train 配另一個比例的 test，指標會同時混進「模型能力」與「分布位移」兩件事；重切成統一的分層比例後，三個 split 的類別分布一致，驗證集與測試集的指標才可以互相解讀。
  2. **明確寫出代價**：重切後的測試集不是文獻回報的那 624 張，所以本專案的數字不能與外部結果比較，只能內部互比——而「內部互比」成立的前提是每組實驗讀同一份切分名單（見 [section_2.md](section_2.md) 的 2.6）。
  3. 保留官方切分這條路。[data/dataset.py:123-143](../data/dataset.py#L123-L143) 的 `build_datasets()` 還在（走官方目錄的 `ImageFolder`），但兩個 notebook 都沒有匯入它——**它目前是為第五章的對照實驗預留的，不是殘留程式碼**。要嘛在第五章真的用它跑一組「官方切分」對照（[section_3.md](section_3.md) 的 E1，也是對照實驗的第一組），要嘛就該刪掉。留著不用又不說明，三週後會分不清它是哪一種。
- 為什麼：第一章的收尾是「所以必須重新設計資料切分」。現在程式做的事比這句話更多，報告如果只講 val，讀者對著 `metadata.json` 裡的 `split_scheme = regrouped_70_15_15` 會立刻發現論述沒有覆蓋實作。
- 驗收：報告中「為什麼重切」的每一個理由，都能指向 `dataset_stats.json` 裡的一個欄位。

## 四、本章報告素材清單

| 素材 | 來源任務 | 檔案 |
| --- | --- | --- |
| 資料集規模與類別分布表 | 1.1 | `dataset_stats.csv` |
| 總張數對帳（5856 vs 5863） | 1.2 | `dataset_stats.json` |
| 官方 val 限制的量化說明 | 1.3 | `dataset_stats.json` 摘要欄位 |
| 官方 train/test 類別比例差距 | 1.3 | `dataset_stats.json` |
| 圖 1-1 原始切分類別分布 | 1.4 | `fig_1_original_split.png` |
| 圖 1-2 樣本影像 | 1.5 | `fig_1_samples.png` |
| 重切決定的理由與代價 | 1.6 | 論述，數字引自 `dataset_stats.json` |
