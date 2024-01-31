import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.models import EfficientNet_V2_L_Weights, efficientnet_v2_l
from tqdm import tqdm


class HuggingFaceImageDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

        columns = set(dataset.column_names)
        if "image" not in columns:
            raise ValueError(f"Expected an 'image' column, found: {dataset.column_names}")
        if "label" not in columns:
            raise ValueError(f"Expected a 'label' column, found: {dataset.column_names}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = self.dataset[index]
        image = row["image"].convert("RGB")
        label = int(row["label"])
        return self.transform(image), label


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_hf_dataset(args):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets\n"
            "Install it with: venv/bin/pip install datasets huggingface_hub"
        ) from exc

    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split=args.split,
        token=args.token,
        trust_remote_code=args.trust_remote_code,
    )

    if args.shuffle:
        dataset = dataset.shuffle(seed=args.seed)
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    return dataset


def evaluate(model, loader, device):
    model.eval()
    top1_correct = 0
    top5_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, pred = outputs.topk(5, dim=1)

            total += labels.size(0)
            top1_correct += (pred[:, 0] == labels).sum().item()
            top5_correct += pred.eq(labels.view(-1, 1)).sum().item()

    return {
        "top1": 100 * top1_correct / total,
        "top5": 100 * top5_correct / total,
    }


def train(args):
    torch.manual_seed(args.seed)
    device = pick_device()
    print("Device:", device)

    weights = EfficientNet_V2_L_Weights.IMAGENET1K_V1
    transform = weights.transforms()

    hf_dataset = load_hf_dataset(args)
    dataset = HuggingFaceImageDataset(hf_dataset, transform)
    print("Total images:", len(dataset))

    val_size = max(1, int(args.val_fraction * len(dataset)))
    train_size = len(dataset) - val_size
    if train_size < 1:
        raise ValueError("Dataset is too small after applying --max-samples")

    train_data, val_data = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device == "cuda",
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device == "cuda",
    )

    model = efficientnet_v2_l(weights=weights)
    if args.freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_top1 = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for images, labels in progress:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        metrics = evaluate(model, val_loader, device)
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}")
        print(f"Loss: {avg_loss:.4f}")
        print(f"Top-1 Accuracy: {metrics['top1']:.2f}%")
        print(f"Top-5 Accuracy: {metrics['top5']:.2f}%")

        if metrics["top1"] >= best_top1:
            best_top1 = metrics["top1"]
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_path)
            print(f"Saved best model: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune EfficientNet V2-L on a Hugging Face ImageNet-1k dataset."
    )
    parser.add_argument("--dataset-name", default="ILSVRC/imagenet-1k")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="efficientnetv2l_hf_imagenet1k_finetuned.pth")
    parser.add_argument("--token", default=None, help="Hugging Face token for gated datasets.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    parser.add_argument("--train-full-model", dest="freeze_backbone", action="store_false")
    parser.set_defaults(shuffle=True, freeze_backbone=True)
    return parser.parse_args()


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
