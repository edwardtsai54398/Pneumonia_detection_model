import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from data.dataset import build_transforms
from models.builder import build_model
from models.inference import predict


def load_metadata(metadata_path: Path):
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Single image inference for pneumonia classifier")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True, help="best_model.pth 路徑")
    parser.add_argument("--metadata_path", type=str, required=True, help="metadata.json 路徑")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata = load_metadata(Path(args.metadata_path))
    idx_to_class = {int(k): v for k, v in metadata["idx_to_class"].items()}
    img_size = int(metadata.get("img_size", 224))
    model_name = metadata.get("model_name")
    num_classes = len(idx_to_class)

    # pretrained=False：checkpoint 的 load_state_dict(strict=True) 會覆蓋所有參數，
    # 推論時不需要先下載 ImageNet 預訓練權重。
    model = build_model(num_classes, device, model_name=model_name, pretrained=False)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    tf = build_transforms(img_size=img_size)["val"]

    image = Image.open(args.image_path).convert("RGB")
    x = tf(image).unsqueeze(0).to(device)

    _, _, scores = predict(model, x, idx_to_class=idx_to_class)
    probs = scores.squeeze(0)
    pred_idx = int(torch.argmax(probs).item())

    pred_label = idx_to_class[pred_idx]
    confidence = float(probs[pred_idx].item())
    print(f"Prediction: {pred_label}, confidence: {confidence:.4f}")
    print("Class probabilities:")
    for i in range(num_classes):
        print(f"  {idx_to_class[i]}: {float(probs[i].item()):.4f}")


if __name__ == "__main__":
    main()
