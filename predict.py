from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from waste_classifier.model import build_resnet18_model
from waste_classifier.utils import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on one image.")
    parser.add_argument("--image-path", type=str, required=True, help="Path to the image you want to classify.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a saved .pth checkpoint.")
    parser.add_argument("--top-k", type=int, default=3, help="Show the top K class predictions.")
    return parser.parse_args()


def build_inference_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def main():
    args = parse_args()
    device = get_device()

    checkpoint = torch.load(args.checkpoint, map_location=device)
    class_names = checkpoint["class_names"]
    image_size = checkpoint.get("image_size", 224)

    model = build_resnet18_model(
        num_classes=len(class_names),
        freeze_backbone=False,
        use_pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    transform = build_inference_transform(image_size=image_size)

    image = Image.open(Path(args.image_path)).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        top_probabilities, top_indices = torch.topk(probabilities, k=min(args.top_k, len(class_names)), dim=1)

    print(f"Prediction for: {args.image_path}")
    for probability, index in zip(top_probabilities[0], top_indices[0]):
        class_name = class_names[index.item()]
        confidence = probability.item() * 100
        print(f"{class_name}: {confidence:.2f}%")


if __name__ == "__main__":
    main()

