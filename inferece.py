import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights
from tqdm import tqdm
import csv
from pathlib import Path


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    weights = EfficientNet_V2_L_Weights.IMAGENET1K_V1

    model = efficientnet_v2_l(weights=weights)
    model = model.to(device)
    model.eval()

    transform = weights.transforms()

    dataset = ImageNet(
        root="./imagenet",
        split="val",
        transform=transform
    )

    prediction_class_names = weights.meta["categories"]
    true_class_names = [names[0] for names in dataset.classes]

    loader = DataLoader(
        dataset,
        batch_size=32,          # GPU can handle larger batch
        shuffle=False,
        num_workers=4,          # Linux supports workers well
        pin_memory=True         # faster CPU -> GPU transfer
    )

    top1_correct = 0
    top5_correct = 0
    total = 0
    predictions = []

    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            _, pred = outputs.topk(5, dim=1)

            total += labels.size(0)

            top1_correct += (pred[:, 0] == labels).sum().item()
            top5_correct += pred.eq(labels.view(-1, 1)).sum().item()

    prediction_class_names = weights.meta["categories"]
    true_class_names = [names[0] for names in dataset.classes]

    top1 = 100 * top1_correct / total
    top5 = 100 * top5_correct / total

    print(f"Top-1 Accuracy: {top1:.3f}%")
    print(f"Top-5 Accuracy: {top5:.3f}%")

    output_path = Path("results/predicted_classes.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["image_path", "true_class", "predicted_class", "top5_classes"]
        )
        writer.writeheader()
        writer.writerows(predictions)

    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()