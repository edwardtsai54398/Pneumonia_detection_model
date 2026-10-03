IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

MODELS = {
    "EFFICIENTNET_B3": "efficientnet_b3",
    "RESNET18": "resnet18",
    "RESNET50": "resnet50",
}
DEFAULT_MODEL_NAME = MODELS["EFFICIENTNET_B3"]

POSITIVE_CLASS_NAME = "PNEUMONIA"

# 類別名稱。必須維持「字母排序」，因為 torchvision 的 ImageFolder 是按字母排序
# 類別目錄來決定索引（NORMAL=0, PNEUMONIA=1）。data/split.py 的索引式切分從這個
# 清單推導 class_to_idx，兩者必須一致——否則舊的 best_model.pth 配新的
# metadata.json 會「安靜地反向預測」：不丟例外、信心值看起來還很合理。
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
if CLASS_NAMES != sorted(CLASS_NAMES):
    raise ValueError(f"CLASS_NAMES 必須是字母排序: {CLASS_NAMES}")

# 資料切分用的種子，刻意與訓練用的 RANDOM_SEED 分開。
# 兩者混用會讓「換 seed 比較穩定性」同時換掉 train/val/test 切分，
# 導致不同 seed 的實驗根本不能互比。
DEFAULT_SPLIT_SEED = 42
