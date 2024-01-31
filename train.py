import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageNet
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights
from tqdm import tqdm


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    weights = EfficientNet_V2_L_Weights.IMAGENET1K_V1
    transform = weights.transforms()

    dataset = ImageNet(
        root="./imagenet",
        split="val",
        transform=transform
    )

    print("Total images:", len(dataset))

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_data, val_data = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        # train_data,
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        # val_data,
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    model = efficientnet_v2_l(weights=weights)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    epochs = 20

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        model.eval()
        top1_correct = 0
        top5_correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, pred = outputs.topk(5, dim=1)

                total += labels.size(0)
                top1_correct += (pred[:, 0] == labels).sum().item()
                top5_correct += pred.eq(labels.view(-1, 1)).sum().item()

        top1 = 100 * top1_correct / total
        top5 = 100 * top5_correct / total

        print(f"Epoch {epoch+1}")
        print(f"Loss: {total_loss / len(train_loader):.4f}")
        print(f"Top-1 Accuracy: {top1:.2f}%")
        print(f"Top-5 Accuracy: {top5:.2f}%")

    torch.save(model.state_dict(), "efficientnetv2l_val_finetuned.pth")
    print("Saved: efficientnetv2l_val_finetuned.pth")


if __name__ == "__main__":
    main()