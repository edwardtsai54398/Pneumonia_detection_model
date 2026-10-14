# Section 5｜實驗結果與分析 — 程式工作清單

> 對應 [agenda.md](agenda.md) 第五章。訓練曲線與混淆矩陣已經有了；**Grad-CAM 是整份專案最大的單一空缺**（連逐樣本預測紀錄都沒有，所以現在連「哪幾張判錯了」都查不出來）。跨實驗比較也還沒有任何程式。

## 一、現有程式碼已覆蓋

| 需求 | 現況 | 位置 |
| --- | --- | --- |
| 訓練曲線 | 2×3 子圖：loss／F1／accuracy／precision／recall／LR，train 與 val 同圖 | [engine/visualize.py:5-41](../engine/visualize.py#L5-L41) |
| 混淆矩陣 | 列正規化熱圖，格內同時標比例與原始計數 | [engine/visualize.py:44-77](../engine/visualize.py#L44-L77) |
| 結果存檔 | best weights、metadata、history、兩張 png 寫入時間戳目錄 | [utils.py:49-94](../utils.py#L49-L94) |
| 單張推論 | CLI 讀 metadata 自動選架構，輸出各類別機率 | [infer_single.py](../infer_single.py) |
| 批次推論 | softmax ＋ argmax，回傳索引／類名／分數 | [models/inference.py](../models/inference.py) |

## 二、缺口

1. **Grad-CAM 完全沒有。** 沒有任何 CAM 相關模組。這是 agenda 第五章第三點的全部內容，也是第六章「要驗證模型關注對不對地方」那個反思的唯一支撐。
2. **沒有逐樣本預測紀錄。** `run_one_epoch()` 只累積統計量，不保留任何樣本層級的資訊，更沒有檔案路徑。所以**無法挑出誤判樣本**，Grad-CAM 的「正確 vs 錯誤樣本比對」就無從下手。這是 Grad-CAM 的前置依賴，要先做。
3. **沒有跨實驗彙總。** 每組實驗各自寫一個 `outputs/<name>_<timestamp>/`，沒有程式把多個 `metadata.json` 讀出來並排比較。第五章第二點（凍結 vs 微調，用數字呈現）現在只能手抄。
4. **ROC 曲線、PR 曲線、門檻掃描曲線都沒有繪圖函式。**
5. **訓練曲線沒有標示關鍵節點。** 解凍發生在哪個 epoch、best checkpoint 在哪個 epoch，圖上看不出來。偏偏「凍結→解凍那一刻指標的跳動」正是特徵遷移有效性最直觀的視覺證據。
6. **混淆矩陣一次只畫一張**，兩種策略要並排比較得自己拼圖。
7. **matplotlib 中文字型問題還沒處理。** 預設字型不含中文，圖表若用中文標題會變成一排豆腐方格。這件事通常在交報告前一晚才被發現。

## 三、要做的程式工作

### 5.1 逐樣本預測匯出（Grad-CAM 的前置作業，優先做）

- 位置：新檔 `engine/evaluate.py`
- 做什麼：對測試集跑一次推論，輸出每張影像一列：相對路徑、真實標籤、正類別機率、預測標籤（套第四章 4.4 選定的門檻）、是否正確、落在 TP/TN/FP/FN 哪一格。
- 實作要注意的兩件事：
  1. **要取得檔案路徑。** `ImageFolder` 把路徑放在 `.samples` 裡，但 DataLoader 預設只回傳 (image, label)。做法有兩種：在 `shuffle=False` 的前提下用批次索引回推 `.samples` 的位置（簡單但很脆弱，一旦誰把 shuffle 打開就會靜默錯位），或自訂 dataset 讓它連路徑一起回傳（推薦，與 [section_2.md](section_2.md) 的 2.6 自訂 dataset 可共用）。
  2. 評估時 DataLoader 必須 `shuffle=False`，否則每次匯出的順序不同、難以比對。
- 產出：輸出目錄下的 `predictions.csv`。
- 為什麼值得做：這份 CSV 一次解決三件事——Grad-CAM 的樣本挑選、第四章 bootstrap CI 的輸入、以及「最沒信心的判斷長什麼樣」的誤差分析。

### 5.2 Grad-CAM 實作

- 位置：新檔 `engine/gradcam.py`
- 做什麼（流程，不含實作細節）：
  1. **選定目標層**：取骨幹最後一個卷積 block 的輸出。EfficientNet_B3 用 `features[-1]`，ResNet 用 `layer4`。因為專案支援多骨幹，要做成「骨幹名稱 → 目標層」的對照表，放在與 [models/builder.py](../models/builder.py) 的 registry 同一處管理。
  2. **掛 hook**：在目標層註冊 forward hook 存下活化值、backward hook 存下梯度。
  3. **反傳目標類別的 logit**：對指定類別（通常是預測類別，分析漏診時也可指定 PNEUMONIA）做反向傳播。
  4. **算權重與熱圖**：梯度對空間維度做全域平均池化當各 channel 的權重，加權後沿 channel 相加，過 ReLU，放大回原圖尺寸，正規化到 0–1。
  5. **疊圖**：以 colormap 上色後與原圖以透明度混合。
- **最容易踩的三個坑**（請務必注意）：
  - **eval 模式下仍然必須開著梯度。** 習慣性地把推論包進 `torch.no_grad()`，backward hook 就永遠拿不到梯度，熱圖會是全黑或直接報錯。Grad-CAM 需要 `model.eval()` ＋ **有** 梯度，這個組合和專案其他地方的推論寫法（[models/inference.py:5](../models/inference.py#L5) 的 `@torch.no_grad()`）正好相反，不要直接複用那條路徑。
  - **疊圖需要未經 Normalize 的影像。** 經過 ImageNet 標準化的 tensor 直接畫出來會是過曝的雜訊。兩個選擇：反標準化 tensor，或依 5.1 的 CSV 裡的路徑**重新讀原圖**。建議後者——原始解析度比 224×224 高，熱圖疊上去清楚得多。
  - **`Grayscale(num_output_channels=3)` 的影響**：三個通道內容相同，梯度會平均分散，這是正常的，不要誤判為實作錯誤。
- 驗收（這一步不要跳）：挑一張模型高信心判對的 PNEUMONIA，熱圖應該落在肺野區域而非影像邊框、角落或文字標記上。**如果熱圖集中在影像邊緣或拍攝註記，那是模型抓到了資料集的拍攝偽影（shortcut learning）**——這個發現本身就是第五章與第六章最有價值的材料，不要當成 bug 想辦法藏掉。

### 5.3 Grad-CAM 對照網格

- 位置：[engine/visualize.py](../engine/visualize.py)
- 做什麼：從 5.1 的 CSV 各取 3–4 張 TP／TN／FP／FN 樣本（FN 最重要——那是漏診），排成四列網格，每格標真實標籤、預測標籤與機率值。
- 建議額外做一組：**同一張影像在凍結模型 vs 微調模型下的熱圖並排**。這會直接呈現「微調後模型的注意力從哪裡移到哪裡」，是比任何指標數字都更有說服力的遷移學習證據。
- 產出：`docs/assets/fig_5_gradcam_grid.png`、`fig_5_gradcam_frozen_vs_finetune.png`。
- 報告怎麼用：agenda 要求「找出模型誤判時實際關注的區域」。FN 樣本的熱圖如果落在正常肺野、或離開病灶區，就能具體寫出模型漏診的機制，而不是只說「模型表現不夠好」。

### 5.4 跨實驗彙總

- 位置：新檔 `experiments/summarize.py`
- 做什麼：掃描 `outputs/*/metadata.json`，組出一張對照表：實驗名稱、骨幹、策略、test 的 recall／precision／specificity／F1／AUC／accuracy、best epoch、參數量、每 epoch 時間。輸出成 markdown 表格（可直接貼進報告）與 CSV。
- 依賴：[section_3.md](section_3.md) 的 3.4 必須先做完，否則 metadata 裡沒有策略與超參數欄位可讀。
- 另加一張分組柱狀圖：以指標為 x 軸、策略為分組，呈現凍結 vs 微調的差距。**若已完成第四章 4.6 的 bootstrap CI，記得畫上誤差棒**——有誤差棒的比較圖，可信度完全是另一個層級。
- 產出：`docs/assets/experiment_comparison.md`、`.csv`、`fig_5_strategy_comparison.png`。

### 5.5 訓練曲線強化

- 位置：[engine/visualize.py:5-41](../engine/visualize.py#L5-L41)
- 做什麼：
  1. 加上垂直參考線：解凍 epoch（虛線）、best epoch（實線），各自加圖例說明。
  2. 版面從 2×3 擴成能容納 AUC 的格數（第四章 4.3 會往 history 加 AUC 欄位）。
  3. 讓函式容忍 history 裡缺少某些欄位（frozen 策略沒有解凍 epoch），缺就不畫該參考線，不要報錯。
- 為什麼：解凍那一刻的指標跳躍是第五章「特徵遷移有效程度」最直接的視覺證據，但要讓讀者看出來，圖上必須標明轉折發生在哪裡。

### 5.6 ROC／PR／門檻掃描曲線

- 位置：[engine/visualize.py](../engine/visualize.py)
- 做什麼：三個新繪圖函式——
  - ROC 曲線（標 AUC 值、對角線基準線，**並在圖上標出 4.4 選定的工作點**）
  - PR 曲線（標 average precision，以及正類別占比作為隨機基準）
  - 門檻掃描曲線（x 軸 threshold，同圖畫 recall／precision／specificity／F1，並在選定門檻處畫垂直線）
- 建議把多組實驗的 ROC 畫在同一張圖，直接比較凍結 vs 微調。
- 產出：`fig_5_roc.png`、`fig_5_pr.png`、`fig_5_threshold_sweep.png`。

### 5.7 混淆矩陣並排

- 位置：[engine/visualize.py:44-77](../engine/visualize.py#L44-L77)
- 做什麼：讓函式支援接收多個混淆矩陣並排成一列（共用同一個色階與 colorbar，否則顏色深淺無法互相比較），每張標策略名稱。
- 產出：`fig_5_cm_comparison.png`。

### 5.8 matplotlib 中文字型（別留到最後）

- 位置：[engine/visualize.py](../engine/visualize.py) 頂部統一設定
- 做什麼：二選一並明確決定——
  - **（建議）圖表標籤全用英文**，中文說明寫在報告正文的圖說裡。零風險、跨機器一致。
  - 或在模組載入時統一指定中文字型（Windows 可用微軟正黑體／標楷體），並設定負號正常顯示。缺點是換到 Colab 跑就會失效，要另外處理字型安裝。
- 為什麼現在就決定：一旦所有圖都產出來才發現標題是豆腐方格，就得全部重跑。

## 四、建議的執行順序

Grad-CAM 這條線有依賴關係，順序弄錯會卡住：

1. **5.8 字型決策**（先定，免得重跑）
2. **5.1 逐樣本預測匯出**（Grad-CAM 與 bootstrap CI 的共同前置）
3. **5.5 ＋ 5.6 ＋ 5.7 繪圖**（可與下一步並行）
4. **5.2 Grad-CAM 核心**（先在單張影像上驗收通過，再往下）
5. **5.3 Grad-CAM 對照網格**
6. **5.4 跨實驗彙總**（要等 [section_3.md](section_3.md) 的實驗都跑完）

## 五、本章報告素材清單

| 素材 | 來源任務 | 檔案 |
| --- | --- | --- |
| 訓練曲線（含解凍／best 標記） | 5.5 | `training_curves.png` |
| 混淆矩陣並排比較 | 5.7 | `fig_5_cm_comparison.png` |
| 策略對照表與柱狀圖 | 5.4 | `experiment_comparison.md`、`fig_5_strategy_comparison.png` |
| ROC／PR／門檻掃描 | 5.6 | `fig_5_roc.png` 等 |
| Grad-CAM 四格對照網格 | 5.3 | `fig_5_gradcam_grid.png` |
| Grad-CAM 凍結 vs 微調 | 5.3 | `fig_5_gradcam_frozen_vs_finetune.png` |
| 誤判樣本清單 | 5.1 | `predictions.csv` |

> 第六章（結論與反思）沒有新的程式工作，但它的三個論點全部依賴本章產物：遷移學習的有效性靠 5.4 的對照表，「不能只看準確率」靠 [section_4.md](section_4.md) 的指標設計與 4.2 的退化解實證，「要驗證關注對不對地方」靠 5.3 的 Grad-CAM。**本章沒做完，第六章就只能寫空話。**
