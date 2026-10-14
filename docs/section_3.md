# Section 3｜模型架構選擇與訓練策略 — 程式工作清單

> 對應 [agenda.md](agenda.md) 第三章。骨幹與訓練迴圈的**能力**都在，缺的是「把對照實驗變成可執行、可彙總、可重現的東西」，以及一個目前會直接傷到論述的設定問題。

## 一、現有程式碼已覆蓋

| 需求 | 現況 | 位置 |
| --- | --- | --- |
| 骨幹候選 | `efficientnet_b3` / `resnet18` / `resnet50`，各自綁 ImageNet 權重，以 registry 管理 | [models/builder.py:14-18](../models/builder.py#L14-L18) |
| 通用換頭 | `replace_head()` 取最後一個 module 的 `in_features` 換成新的 `Linear` | [models/builder.py:21-29](../models/builder.py#L21-L29) |
| 骨幹凍結 | `freeze_backbone=True` 時關掉非 `classifier`／非 `fc` 參數的梯度 | [models/builder.py:40-43](../models/builder.py#L40-L43) |
| 階段式解凍 | 到 `unfreeze_epoch` 解凍骨幹，以 `lr * backbone_lr_factor` 加成第二個 param group | [engine/trainer.py:131-139](../engine/trainer.py#L131-L139) |
| 優化器／排程 | AdamW（只收 `requires_grad` 的參數）＋ `ReduceLROnPlateau` | notebook Cell 4 |
| 資源成本量測 | `num_parameters`（M）、`train_time_per_epoch`（秒） | [utils.py:64](../utils.py#L64)、[utils.py:24-39](../utils.py#L24-L39) |

## 二、缺口

1. **訓練策略沒有明確的實驗模式開關，而且目前的預設值不是 agenda 要對照的兩組之一。** 現在 `FREEZE_BACKBONE=True` ＋ `UNFREEZE_EPOCH=4` 跑出來的是**階段式解凍**（第三種策略）。agenda 要對照的是「凍結骨幹」vs「全微調」。更麻煩的是：想跑「全程凍結」，現在只能靠把 `unfreeze_epoch` 設得比 `epochs` 大來迂迴達成——這種隱性約定在報告裡無法交代，也很容易自己弄錯。
2. **沒有實驗 runner，也沒有結果彙總。** 要換策略得手改 notebook 的 Cell 2 再重跑，結果散落在各個時間戳目錄，沒有任何程式把它們合起來比較。
3. **`metadata.json` 幾乎沒有記錄超參數。** 目前只存 `model_name`、`num_parameters`、`class_to_idx`、`img_size`、`best_val_f1`、test 指標、每 epoch 時間（[utils.py:68-77](../utils.py#L68-L77)）。**LR、batch size、weight decay、epochs、seed、freeze 設定、unfreeze epoch、backbone_lr_factor、patience 全都沒存。** 兩個後果：附錄「超參數設定細節」沒有可信來源；三週後看到一個 outputs 目錄，你無法知道它是哪一組設定跑出來的。
4. **`replace_head()` 有個隱性假設**：它拿 `named_modules()` 的最後一項當分類層。對現有三個骨幹成立，但這是「剛好」而非保證——若要擴充骨幹候選（DenseNet、ViT 之類），最後一個 module 可能是 dropout 或 norm，會直接換錯層且不會報錯。
5. **資源成本量測不足以支撐「運算資源限制」的篩選依據。** 只有參數量與每 epoch 時間，缺峰值顯存與推論延遲——而顯存才是實際決定「這個骨幹能不能在手上的 GPU 跑」的那個數字。
6. 沒有混合精度（AMP）。非必要，但若受限於 Colab 的 GPU 時數，這是成本最低的加速手段。

## 三、要做的程式工作

### 3.1 把訓練策略收成一個明確的模式參數

- 位置：[engine/trainer.py](../engine/trainer.py) 的 `train_model()` ＋ notebook Cell 2
- 做什麼：以一個策略名稱取代現在三個互相牽連的布林／數字參數，至少涵蓋三種模式：
  - **frozen**：骨幹全程凍結，只訓練分類頭，不解凍。
  - **finetune**：第 0 epoch 起全部參數一起訓練，單一 learning rate。
  - **staged**：現行行為（先凍結，到指定 epoch 解凍並給骨幹較小的 lr）。
- 內部仍可沿用現有的 `should_freeze_backbone` / `unfreeze_epoch` 邏輯，但由策略名稱推導出這些值，**不要讓使用者自己湊**。`frozen` 模式要走明確的「不解凍」分支，而不是靠把 unfreeze epoch 設成超過總 epoch 數。
- 連帶要做：策略名稱寫進 metadata 與輸出目錄名稱，讓目錄名自己就能說明它是哪一組。
- 驗收：三種模式各跑 2 個 epoch 的 smoke test，檢查 `requires_grad` 為 True 的參數數量與 optimizer 的 param group 數量符合預期（frozen 全程 1 組、finetune 全程 1 組但參數量大、staged 在解凍後變 2 組）。

### 3.2 對照實驗 runner

- 位置：新檔 `experiments/run_comparison.py`（或等價的 notebook cell）
- 做什麼：吃一份實驗設定清單（每項＝策略 × 骨幹 × 其他超參數），依序建模、訓練、評估、存檔，每一組寫進獨立的 outputs 子目錄。其他條件（seed、切分清單、batch size、img_size、epochs 上限）必須強制一致。
- 為什麼：第五章要「用數字呈現特徵遷移的有效程度」。一次手動改參數重跑，過程中很難保證沒有順手動到別的東西；跑完也難證明當初真的只差一個變因。
- 注意：**切分清單要在第一組實驗產生後固定下來，之後每組都讀同一份**（見 [section_2.md](section_2.md) 的 2.6），不要每組各自重切。即使 seed 相同，切分邏輯一改就會錯開。
- 建議至少跑完的組合。**這五組不是五個獨立實驗，而是一條「只差一個變因」的對照鏈**，編號順序即執行順序，也就是報告的敘事順序——從官方設定出發、發現問題、改掉它、再逐項檢驗改法：

  | 編號 | 骨幹 | 對照組 | 變因 |
  | --- | --- | --- | --- |
  | E1 | efficientnet_b3 | 起點 | 切分改為重切 70/15/15、骨幹全程凍結，只訓練分類頭 |
  | E2 | efficientnet_b3 | E1 | 第 4 epoch 解凍骨幹一起訓練 |
  | E3 | efficientnet_b3 | E2、E1 | 第 0 epoch 起全部參數一起訓練 |
  | E4 | resnet18 | E1 | 骨幹換成小一個量級的 resnet18 |


#### 執行順序

編號即順序，E1 先跑。除了敘事上它是起點之外，還有兩個實際理由：**它是唯一不依賴 [section_2.md](section_2.md) 2.6（切分名單匯出）的一組**，所以 2.6 還沒做完時它就能開跑；而 `build_datasets()` 是目前整個 repo 裡**沒有被執行過的程式**（兩個 notebook 都沒匯入它），transform 參數、回傳結構、與 `make_dataloaders()` 的介面都沒驗證過——它可能根本跑不起來，這件事要在第一天發現，不是在最後一天。

E2–E5 必須讀同一份凍結的切分名單。E5 換的是骨幹，與 E2–E4 互不影響，GPU 若排得開可以並行。時間不夠時能砍的只有 E5；E1–E4 缺任一組，對照鏈就斷一環。

**但下面三項必須在 E1 之前做完，它們一旦後補就得整批重跑：**

1. **[section_4.md](section_4.md) 的 4.1 ＋ 4.2（監控指標切到 Recall ＋ checkpoint 守門）。** 這是最硬的一道門。監控指標決定**哪一個 epoch 的權重被存成 best**，而現在預設還是 F1。等五組跑完才換成 Recall，存下來的 checkpoint 全都是按舊規則選的，無法代表報告所宣稱的「以 Recall 為準」——只能全部重跑。4.2 的守門要一起做，否則換成 Recall 之後存下來的可能是全預測 PNEUMONIA 的退化解。
2. **3.1 策略開關。** 不先收成明確的 `frozen` / `finetune` / `staged`，E3 就只能靠「把 unfreeze epoch 設得比 epochs 大」迂迴達成，而這件事在 metadata 裡看不出來。
3. **3.4 metadata 寫入完整設定。** 跑完才補，先跑的那幾組就沒有欄位可讀，[section_5.md](section_5.md) 的 5.4 彙總表拼不出來。

### 3.3 骨幹候選的量化比較表

- 位置：新檔 `experiments/profile_backbones.py`
- 做什麼：對三個骨幹各自量測並輸出一張表：總參數量、可訓練參數量（frozen 模式下）、**峰值顯存**（以 `torch.cuda.max_memory_allocated` 在跑過一個 batch 的前後向之後取值，記得先 reset）、單 batch 前向時間、單張推論延遲。不需要完整訓練。
- 為什麼：agenda 說骨幹篩選依據是「文獻回顧＋運算資源限制」。文獻那半是寫作，**運算資源這半必須是自己機器上的實測數字**，否則「因資源限制而選 B3」只是說法。
- 產出：`docs/assets/backbone_profile.csv`，第三章表 3-1。

### 3.4 把完整實驗設定寫進 metadata（同時服務附錄）

- 位置：[utils.py](../utils.py) 的 `save_results()`
- 做什麼：改成接受一個「實驗設定」字典並整包存進 `metadata.json`，至少包含：骨幹名稱、策略、img_size、batch size、epochs 上限、實際訓練到第幾 epoch、optimizer 名稱、LR、weight decay、backbone_lr_factor、unfreeze epoch、scheduler 設定（factor／patience／mode）、early stopping 的 patience 與 min_delta、監控指標名稱、seed、class weight 實際數值、切分模式與比例、是否啟用 sampler／AMP、裝置與 GPU 型號。
- 為什麼：這一項同時解決三件事——附錄的超參數表格可以直接由 metadata 生成、3.2 的彙總表有欄位可讀、實驗可重現。**投入很小，是整份清單裡性價比最高的一項。**
- 順帶修一個命名問題：欄位目前叫 `best_val_f1`（[utils.py:74](../utils.py#L74)），但第四章要把監控指標換成 Recall（見 [section_4.md](section_4.md)）。改成通用的「監控指標名稱 ＋ 該指標最佳值」兩個欄位，否則報告裡會出現一個叫 f1 、實際存 recall 的數字。

### 3.5 `replace_head()` 的穩健性

- 位置：[models/builder.py:21-29](../models/builder.py#L21-L29)
- 做什麼：兩個選擇——(a) 在 registry 裡明確記下每個骨幹的分類層屬性路徑（B3 是 `classifier.1`、ResNet 是 `fc`），(b) 保留自動尋找但加檢查，確認取到的 module 是 `nn.Linear` 且有 `in_features`，否則拋錯。
- 為什麼：現在換錯層時不會報錯，只會安靜地訓練出一個壞模型。以目前只有三個骨幹來說不急，但如果要加第四個候選就必須先處理。
- 附帶可考慮：讓分類頭支援加一層 dropout。B3 原本的 `classifier` 就含 dropout，現在整個被換掉了——這點值得在報告裡提一句，說明有意識地做了這個取捨。

### 3.6 解凍邏輯的一個細節確認

- 位置：[engine/trainer.py:134-137](../engine/trainer.py#L134-L137)
- 現況：解凍條件是「名稱不含 `classifier`」且「目前 `requires_grad` 為 False」。對 ResNet（分類頭叫 `fc`）之所以不會把 `fc` 重複加進新 param group，是因為第二個條件擋住了——`fc` 本來就 `requires_grad=True`。
- 要做什麼：這個行為是對的，但它的正確性依賴參數的當前狀態，而不是依賴一個明確的「這是分類頭」判斷。建議改成與 `freeze_backbone` 同一組判準（同時檢查 `classifier` 與 `fc`），讓兩處的「什麼算骨幹」只有一個定義。
- 驗收：在 ResNet ＋ staged 模式下解凍後，檢查 optimizer 的兩個 param group 沒有重複參數（重複會讓那些參數被套用兩次更新）。

### 3.7 混合精度（選配）

- 位置：[engine/trainer.py](../engine/trainer.py) 的 `run_one_epoch()`
- 做什麼：以 autocast ＋ GradScaler 包住前後向，做成可開關的參數，預設關閉。
- 為什麼：若 Colab GPU 時數是瓶頸，這能縮短每 epoch 時間並降低顯存，讓 3.2 的五組實驗跑得完。開啟與否要寫進 metadata，因為它會輕微影響數值結果。

## 四、本章報告素材清單

| 素材 | 來源任務 | 檔案 |
| --- | --- | --- |
| 表 3-1 骨幹候選量化比較 | 3.3 | `backbone_profile.csv` |
| 三種訓練策略的定義與差異說明 | 3.1 | 程式中的策略定義 |
| 附錄 超參數設定細節 | 3.4 | 各 `metadata.json` |
| 實驗矩陣（跑了哪幾組、條件如何對齊） | 3.2 | 實驗設定清單 |
