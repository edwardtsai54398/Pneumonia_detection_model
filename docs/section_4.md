# Section 4｜評估指標與模型選擇策略 — 程式工作清單

> 對應 [agenda.md](agenda.md) 第四章。章節的核心主張是「**以 Recall 主導模型選擇**」，但程式現在是用 F1 主導、AUC-ROC 根本沒算、決策門檻固定在 0.5。這一章的論述與實作目前是脫鉤的，必須對上。

## 一、現有程式碼已覆蓋

| 需求 | 現況 | 位置 |
| --- | --- | --- |
| 混淆矩陣四格 | 依 `positive_index` 累積 TP/TN/FP/FN | [engine/trainer.py:14-21](../engine/trainer.py#L14-L21) |
| 基本指標 | accuracy、precision、recall、F1 | [engine/trainer.py:24-41](../engine/trainer.py#L24-L41) |
| 2×2 混淆矩陣 | 以 `true*2+pred` 攤平後 bincount，與 `positive_index` 無關 | [engine/trainer.py:85-86](../engine/trainer.py#L85-L86) |
| 監控指標可切換 | `learning_depend_metric` 參數，並自動由名稱推導 min/max 模式 | [engine/trainer.py:106](../engine/trainer.py#L106)、[:122-123](../engine/trainer.py#L122-L123) |
| checkpoint 選擇 | 監控指標改善超過 `min_delta` 才存 best state | [engine/trainer.py:166-174](../engine/trainer.py#L166-L174) |
| early stopping | patience 計數，連續未改善即中止 | [engine/trainer.py:178-180](../engine/trainer.py#L178-L180) |
| **checkpoint 守門鉤子** | `checkpoint_guard` 參數已預留，可在「指標改善」之後再否決一次 | [engine/trainer.py:111](../engine/trainer.py#L111)、[:168-169](../engine/trainer.py#L168-L169) |

> `checkpoint_guard` 已經設計好但**從未被使用**。它正是本章 4.2 要填的那個洞。

## 二、缺口

1. **監控指標預設是 F1，不是 Recall。** notebook 沒有覆寫 `learning_depend_metric`，所以 checkpoint 與 early stopping 現在都看 F1。與 agenda 的主張直接矛盾。
2. **Recall 主導會引入退化解，目前沒有任何防護。** 把全部樣本都預測成 PNEUMONIA 就能拿到 recall = 1.0。在一個正類別占七成的資料集上，這個退化解的 accuracy 還有 0.74、F1 還有 0.85，**看起來不像壞模型**。如果直接把監控指標換成 recall 而不加守門，很可能最後存下來的 best checkpoint 就是這種模型，而且不容易察覺。
3. **AUC-ROC 完全沒有實作。** `run_one_epoch()` 只累積計數，從不保留每筆樣本的機率（[engine/trainer.py:77-86](../engine/trainer.py#L77-L86)），所以現在的資料結構根本算不出 AUC。
4. **決策門檻被硬綁在 argmax（等價於 0.5）。** `binary_metrics_from_logits()` 直接取 argmax（[engine/trainer.py:15](../engine/trainer.py#L15)）。但本章整個論點就是「偽陰性代價高於偽陽性」——如果承認這個不對稱，卻用對稱的 0.5 門檻做決策，論述自己打自己。門檻調整是本章最能體現「代價導向」的具體作為。
5. **缺 specificity 與 NPV。** 討論漏診／誤報代價時，光有 precision/recall 講不完整：specificity 回答「正常人有多少被誤判為肺炎」，NPV 回答「被判為正常的人裡有多少其實是肺炎」——後者才是漏診風險的直接表述。
6. **測試集指標沒有不確定度。** 第五章要說「微調比凍結好」，但 test 只有 624 張，兩組差 1 個百分點可能只是雜訊。
7. **scheduler 的 mode 與監控指標分兩處各寫一份。** notebook 寫死 `ReduceLROnPlateau(mode="max")`，trainer 則自己從指標名稱推導 min/max（[engine/trainer.py:122](../engine/trainer.py#L122)）。只要哪天把監控指標換成 loss 類，兩邊就會不一致，而且**不會報錯**，只會讓 LR 在錯誤的時機衰減。
8. 沒有記錄 best epoch 落在第幾個 epoch、是否觸發 early stopping。

## 三、要做的程式工作

### 4.1 把監控指標切到 Recall

- 位置：notebook Cell 2／實驗設定 ＋ [engine/trainer.py](../engine/trainer.py)
- 做什麼：把監控指標明確設為 `recall`，並寫進 metadata（見 [section_3.md](section_3.md) 的 3.4）。同時把 scheduler 的 mode 改為**由監控指標推導**、不要在 notebook 另寫一份（解掉缺口 7）——最乾淨的做法是讓 trainer 負責建立或至少驗證 scheduler 的 mode 與自己的 `monitor_mode` 一致，不符就拋錯。
- 報告怎麼用：第四章要寫的是「為什麼」。先給錯誤代價的不對稱性（漏診→延誤治療，誤報→多一次檢查），再說明因此 checkpoint 與 early stopping 都以 Recall 為準、Accuracy 僅作參考。

### 4.2 實作 checkpoint 守門條件（本章最關鍵）

- 位置：新增一個 guard 函式，透過既有的 `checkpoint_guard` 參數傳入 `train_model()`
- 做什麼：在「recall 有改善」之後再加一道條件，否則不接受為 best。可用的設計（擇一或組合）：
  - 要求 precision（或 specificity）不低於一個下限，例如 0.80
  - 要求 F1 不比目前最佳 F1 掉超過某個容忍值
  - 直接排除預測極度偏斜的 epoch（例如 FP 數量超過負類別總數的某個比例）
- 為什麼：這一段是第四章**最有價值的內容**。它誠實地呈現「選了 Recall 當主要指標 → 發現它有退化解 → 加上約束條件」的推理過程，而不是只宣稱「我們重視 Recall」。報告裡可以把守門條件寫成一句明確的模型選擇準則：**在 precision 不低於門檻的前提下，最大化 Recall**。
- 驗收：刻意跑一次不加守門的 recall 主導訓練，把它存下來當對照——如果真的出現退化解，這就是報告裡最好的證據圖表；如果沒出現，也要寫明「守門條件未被觸發」，這同樣是結果。

### 4.3 收集逐樣本機率，實作 AUC-ROC 與 PR-AUC

- 位置：[engine/trainer.py](../engine/trainer.py) 的 `run_one_epoch()`
- 做什麼：
  1. 在迴圈內把每個 batch 的正類別 softmax 機率與真實標籤累積起來（train 模式可選擇略過以省記憶體，eval 模式必收）。
  2. epoch 結束後計算 AUC-ROC 與 **PR-AUC（average precision）**，併入回傳的 metrics。
  3. 把 AUC 寫進 `history`（[engine/trainer.py:146-159](../engine/trainer.py#L146-L159) 的 dict 要加欄位），才能畫逐 epoch 的 AUC 曲線。
- 實作注意：
  - 累積的張量要先 `detach()` 並搬到 CPU，否則整個 epoch 的計算圖都會被留住，顯存會爆。
  - AUC 可以用 `sklearn.metrics`（需要加進 [requirements.txt](../requirements.txt)，目前沒有 scikit-learn），或自己用排序實作。用 sklearn 比較省事，也順便能拿到 ROC 曲線的座標點給第五章畫圖。
- 報告怎麼用：AUC-ROC 是**門檻無關**的指標，正好用來回答「換門檻之後指標變好，是模型真的更好還是只是挪了門檻」——AUC 不動而 recall 上升，說明是門檻取捨而非能力提升。這個區分讓第四章的論述完整。

### 4.4 可調決策門檻 ＋ 門檻掃描

- 位置：[engine/trainer.py](../engine/trainer.py) 的 `binary_metrics_from_logits()` ＋ 新增一個門檻選擇函式
- 做什麼：
  1. 讓指標計算支援「以正類別機率 ≥ threshold 判為正」，threshold 預設 0.5（等價於現行 argmax 行為，不破壞既有結果）。
  2. 新增門檻掃描：在**驗證集**上掃 0.05～0.95，記錄每個門檻的 recall／precision／specificity／F1，依照 4.2 的準則（precision 不低於下限時最大化 recall）選出最佳門檻。
  3. 選定的門檻寫進 metadata，**測試集評估時沿用該門檻**。
- 絕對不能做的事：**不要在測試集上選門檻。** 門檻是模型決策的一部分，在 test 上挑選就是資訊洩漏，第五章的數字會失去意義。要在報告裡明確寫出「門檻由驗證集決定」。
- 產出：門檻掃描結果存檔，供第五章畫 threshold sweep 曲線。

### 4.5 補齊指標

- 位置：[engine/trainer.py:24-41](../engine/trainer.py#L24-L41) 的 `summarize_metrics()`
- 做什麼：加上 specificity、NPV、balanced accuracy。四格數字都在手上，只是沒算。
- 順帶留意一個已知行為：precision / recall 的分母目前用 `max(denominator, 1)` 處理（[engine/trainer.py:27-28](../engine/trainer.py#L27-L28)），分母為 0 時回 0 而非未定義。這在正常訓練不會發生，但若報告裡出現 `0.0000`，要知道它可能代表「該類別完全沒有預測」而不是「表現極差」。

### 4.6 測試集指標的信賴區間

- 位置：新檔 `engine/evaluate.py` 或併入彙總腳本
- 做什麼：拿 4.3 收集到的逐樣本機率與標籤，做 bootstrap 重抽（例如 1000 次，每次抽與原本同樣大小、可重複），算出 recall／F1／AUC 的 95% 區間。
- 為什麼：讓第五章可以寫「微調的 recall 為 0.97（95% CI 0.95–0.99），凍結為 0.92（0.89–0.95）」。有區間的比較才能說差距是否可靠；兩個區間大幅重疊時，誠實地說「差異不顯著」比硬講贏了更有價值。

### 4.7 記錄訓練過程的關鍵節點

- 位置：[engine/trainer.py](../engine/trainer.py) 的回傳值 ＋ [utils.py](../utils.py) 的 `save_results()`
- 做什麼：回傳並儲存 best epoch 的索引、是否觸發 early stopping、實際跑完幾個 epoch。
- 為什麼：第五章的訓練曲線要在 best epoch 與解凍 epoch 畫參考線（見 [section_5.md](section_5.md) 的 5.5），需要這些數字；附錄也要交代每組實驗實際訓練長度。

## 四、本章報告素材清單

| 素材 | 來源任務 | 說明 |
| --- | --- | --- |
| 錯誤代價不對稱的指標選擇論述 | 4.1 | 需搭配 specificity／NPV 的實際數字 |
| 模型選擇準則（含守門條件） | 4.2 | 寫成一句可檢驗的規則 |
| 退化解的實證（或未觸發的說明） | 4.2 | 無守門 vs 有守門的對照 |
| AUC-ROC / PR-AUC | 4.3 | 逐 epoch 曲線 ＋ 最終值 |
| 選定的決策門檻與依據 | 4.4 | 門檻掃描表，註明由驗證集決定 |
| 測試集指標的 95% CI | 4.6 | 支撐第五章的策略比較 |
