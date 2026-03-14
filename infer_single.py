import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms


def load_metadata(metadata_path: Path):
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_model(model_name: str, num_classes: int, model_path: Path, device: torch.device):
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
    elif model_name == "resnet50":
        model = models.resnet50(weights=None)
    else:
        raise ValueError(f"不支援的模型: {model_name}")

    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Single image inference for pneumonia classifier")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default=r"C:\Users\Edward\Desktop\Classifier\outputs\best_model.pth")
    parser.add_argument("--metadata_path", type=str, default=r"C:\Users\Edward\Desktop\Classifier\outputs\metadata.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata = load_metadata(Path(args.metadata_path))
    idx_to_class = {int(k): v for k, v in metadata["idx_to_class"].items()}
    img_size = int(metadata.get("img_size", 224))
    model_name = metadata.get("model_name", "resnet18")
    num_classes = len(idx_to_class)

    tf = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    model = build_model(model_name, num_classes, Path(args.model_path), device)

    image = Image.open(args.image_path).convert("RGB")
    x = tf(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        pred_idx = int(torch.argmax(probs).item())

    pred_label = idx_to_class[pred_idx]
    confidence = float(probs[pred_idx].item())
    print(f"Prediction: {pred_label}, confidence: {confidence:.4f}")
    print("Class probabilities:")
    for i in range(num_classes):
        print(f"  {idx_to_class[i]}: {float(probs[i].item()):.4f}")


if __name__ == "__main__":
    main()

