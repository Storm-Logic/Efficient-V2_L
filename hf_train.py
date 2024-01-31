import argparse
import json
import os
from pathlib import Path
from urllib.request import urlretrieve

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.models import EfficientNet_V2_L_Weights, efficientnet_v2_l
from tqdm import tqdm


IMAGENET_CLASS_INDEX_URL = (
    "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
)


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


def load_hf_dataset(args, split, shuffle=False, max_samples=None):
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
        split=split,
        token=args.token,
        trust_remote_code=args.trust_remote_code,
    )

    if shuffle:
        dataset = dataset.shuffle(seed=args.seed)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    return dataset


def load_wnid_to_imagenet_index(class_index_path):
    class_index_path = Path(class_index_path)
    if not class_index_path.exists():
        print(f"Downloading ImageNet class index: {class_index_path}")
        urlretrieve(IMAGENET_CLASS_INDEX_URL, class_index_path)

    with class_index_path.open("r", encoding="utf-8") as file:
        class_index = json.load(file)

    return {
        wnid: int(index)
        for index, (wnid, _class_name) in class_index.items()
    }


def load_custom_imagefolder(args, transform):
    dataset = ImageFolder(args.custom_train_dir, transform=transform)
    wnid_to_imagenet_index = load_wnid_to_imagenet_index(args.imagenet_class_index)
    missing = sorted(set(dataset.classes) - set(wnid_to_imagenet_index))
    if missing:
        raise ValueError(
            "Missing ImageNet label mapping for local folders: "
            + ", ".join(missing)
            + f"\nCheck that {args.imagenet_class_index} is a valid ImageNet class-index JSON."
        )

    local_idx_to_imagenet_idx = {
        local_idx: wnid_to_imagenet_index[wnid]
        for wnid, local_idx in dataset.class_to_idx.items()
    }
    dataset.samples = [
        (path, local_idx_to_imagenet_idx[label])
        for path, label in dataset.samples
    ]
    dataset.imgs = dataset.samples
    dataset.targets = [label for _, label in dataset.samples]
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

    train_hf_dataset = load_hf_dataset(
        args,
        split=args.train_split,
        shuffle=args.shuffle,
        max_samples=args.max_samples,
    )
    val_hf_dataset = load_hf_dataset(
        args,
        split=args.validation_split,
        shuffle=False,
        max_samples=args.max_validation_samples,
    )
    train_data = HuggingFaceImageDataset(train_hf_dataset, transform)
    train_parts = [train_data]

    custom_train_data = None
    if args.use_custom_train_data:
        custom_train_data = load_custom_imagefolder(args, transform)
        train_parts.append(custom_train_data)

    combined_train_data = ConcatDataset(train_parts)

    val_data = HuggingFaceImageDataset(val_hf_dataset, transform)
    print("Hugging Face training images:", len(train_data))
    if custom_train_data is not None:
        print("Custom training images:", len(custom_train_data))
    print("Total training images:", len(combined_train_data))
    print("Validation images:", len(val_data))

    train_loader = DataLoader(
        combined_train_data,
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

    model = efficientnet_v2_l(weights=None)
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
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--max-validation-samples", type=int, default=100)
    parser.add_argument("--custom-train-dir", default="val")
    parser.add_argument("--imagenet-class-index", default="imagenet_class_index.json")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="efficientnetv2l_hf_imagenet1k_finetuned.pth")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="Hugging Face token for gated datasets.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-custom-train-data", dest="use_custom_train_data", action="store_false")
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    parser.add_argument("--train-full-model", dest="freeze_backbone", action="store_false")
    parser.set_defaults(shuffle=True, freeze_backbone=True, use_custom_train_data=True)
    return parser.parse_args()


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
