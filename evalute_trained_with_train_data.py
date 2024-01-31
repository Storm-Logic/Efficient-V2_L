import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlretrieve

import torch
from torch.utils.data import DataLoader, Dataset
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
        return self.transform(image), label, f"huggingface:{index}"


class ImageFolderWithPaths(ImageFolder):
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        path = self.samples[index][0]
        return image, label, path


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_hf_validation(args, transform):
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
        split=args.hf_split,
        token=args.token,
        trust_remote_code=args.trust_remote_code,
    )

    if args.hf_max_samples is not None:
        dataset = dataset.select(range(min(args.hf_max_samples, len(dataset))))

    return HuggingFaceImageDataset(dataset, transform)


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


def load_custom_validation(args, transform):
    dataset = ImageFolderWithPaths(args.custom_val_dir, transform=transform)
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


def build_default_model(weights, device):
    model = efficientnet_v2_l(weights=weights)
    model = model.to(device)
    model.eval()
    return model


def build_finetuned_model(checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Fine-tuned checkpoint not found: {checkpoint_path}")

    model = efficientnet_v2_l(weights=None)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    return model


def evaluate_dataset(name, model, dataset, categories, args, device):
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device == "cuda",
    )

    top1_correct = 0
    top5_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, paths in tqdm(loader, desc=name):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, pred = outputs.topk(5, dim=1)

            total += labels.size(0)
            top1_correct += (pred[:, 0] == labels).sum().item()
            top5_correct += pred.eq(labels.view(-1, 1)).sum().item()

            if args.print_predictions:
                for path, true_index, top5_indices in zip(paths, labels.cpu().tolist(), pred.cpu().tolist()):
                    top5_names = [categories[index] for index in top5_indices]
                    print(
                        f"{path}: predicted={top5_names[0]} | "
                        f"true={categories[true_index]} | top5={'; '.join(top5_names)}"
                    )

    top1 = 100 * top1_correct / total
    top5 = 100 * top5_correct / total
    print(f"\n{name}")
    print(f"Images: {total}")
    print(f"Top-1 Accuracy: {top1:.3f}%")
    print(f"Top-5 Accuracy: {top5:.3f}%")
    return {
        "name": name,
        "images": total,
        "top1_accuracy": top1,
        "top5_accuracy": top5,
    }


def save_results(result_path, result_key, results):
    result_path = Path(result_path)
    if result_path.exists() and result_path.stat().st_size > 0:
        with result_path.open("r", encoding="utf-8") as file:
            all_results = json.load(file)
    else:
        all_results = {}

    if "results" in all_results and result_key not in all_results:
        all_results = {"previous_result": all_results}

    all_results[result_key] = results
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8") as file:
        json.dump(all_results, file, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate default and fine-tuned EfficientNet V2-L models."
    )
    parser.add_argument("--models", choices=["both", "default", "finetuned"], default="both")
    parser.add_argument("--checkpoint", default="efficientnetv2l_hf_imagenet1k_finetuned.pth")
    parser.add_argument("--eval-datasets", choices=["both", "hf", "custom"], default="both")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--print-predictions", action="store_true")
    parser.add_argument("--result-json", default="result.json")
    parser.add_argument("--result-key", default="evalute_trained_with_train_data")

    parser.add_argument("--dataset-name", default="ILSVRC/imagenet-1k")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--hf-max-samples", type=int, default=1000)
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--trust-remote-code", action="store_true")

    parser.add_argument("--custom-val-dir", default="val")
    parser.add_argument("--imagenet-class-index", default="imagenet_class_index.json")
    return parser.parse_args()


def main():
    args = parse_args()
    device = pick_device()
    print("Device:", device)

    weights = EfficientNet_V2_L_Weights.IMAGENET1K_V1
    transform = weights.transforms()
    categories = weights.meta["categories"]
    results = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "models": args.models,
        "eval_datasets": args.eval_datasets,
        "checkpoint": args.checkpoint,
        "hf_dataset_name": args.dataset_name,
        "hf_split": args.hf_split,
        "hf_max_samples": args.hf_max_samples,
        "custom_val_dir": args.custom_val_dir,
        "batch_size": args.batch_size,
        "results": [],
    }

    datasets = []

    if args.eval_datasets in {"both", "hf"}:
        hf_dataset = load_hf_validation(args, transform)
        datasets.append((f"Hugging Face ImageNet {args.hf_split}", hf_dataset))

    if args.eval_datasets in {"both", "custom"}:
        custom_dataset = load_custom_validation(args, transform)
        datasets.append(("Custom val folder", custom_dataset))

    if args.models in {"both", "default"}:
        model = build_default_model(weights, device)
        for dataset_name, dataset in datasets:
            result = evaluate_dataset(f"Default model - {dataset_name}", model, dataset, categories, args, device)
            result["model"] = "default"
            result["dataset"] = dataset_name
            results["results"].append(result)
        del model

    if args.models in {"both", "finetuned"}:
        model = build_finetuned_model(args.checkpoint, device)
        for dataset_name, dataset in datasets:
            result = evaluate_dataset(f"Fine-tuned model - {dataset_name}", model, dataset, categories, args, device)
            result["model"] = "finetuned"
            result["dataset"] = dataset_name
            results["results"].append(result)
        del model

    save_results(args.result_json, args.result_key, results)
    print(f"\nSaved results to {args.result_json} under '{args.result_key}'")


if __name__ == "__main__":
    main()
