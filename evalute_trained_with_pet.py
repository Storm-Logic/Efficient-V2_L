import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import OxfordIIITPet
from torchvision.models import EfficientNet_V2_L_Weights, efficientnet_v2_l
from tqdm import tqdm


PET_NUM_CLASSES = 37


class OxfordIIITPetWithPaths(OxfordIIITPet):
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        return image, label, str(self._images[index])


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_pet_dataset(args, transform):
    return OxfordIIITPetWithPaths(
        root=args.pet_root,
        split=args.pet_split,
        target_types="category",
        transform=transform,
        download=args.download_pet,
    )


def replace_classifier(model, num_classes):
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def build_default_model(weights, device):
    model = efficientnet_v2_l(weights=weights)
    model = replace_classifier(model, PET_NUM_CLASSES)
    model = model.to(device)
    model.eval()
    return model, PET_NUM_CLASSES


def build_finetuned_model(checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Fine-tuned checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    classifier_key = "classifier.1.weight"
    if classifier_key not in checkpoint:
        raise KeyError(f"Checkpoint does not contain {classifier_key}")

    num_classes = checkpoint[classifier_key].shape[0]
    model = efficientnet_v2_l(weights=None)
    model = replace_classifier(model, num_classes)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    return model, num_classes


def evaluate_dataset(name, model, dataset, target_categories, output_categories, args, device):
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
    metrics_compatible = len(target_categories) == len(output_categories)

    with torch.no_grad():
        for images, labels, paths in tqdm(loader, desc=name):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            k = min(5, outputs.shape[1])
            _, pred = outputs.topk(k, dim=1)

            total += labels.size(0)
            if metrics_compatible:
                top1_correct += (pred[:, 0] == labels).sum().item()
                top5_correct += pred.eq(labels.view(-1, 1)).sum().item()

            if args.print_predictions:
                for path, true_index, top_indices in zip(paths, labels.cpu().tolist(), pred.cpu().tolist()):
                    top_names = [output_categories[index] for index in top_indices]
                    print(
                        f"{path}: predicted={top_names[0]} | "
                        f"true={target_categories[true_index]} | top{k}={'; '.join(top_names)}"
                    )

    print(f"\n{name}")
    print(f"Images: {total}")
    if metrics_compatible:
        top1 = 100 * top1_correct / total
        top5 = 100 * top5_correct / total
        print(f"Top-1 Accuracy: {top1:.3f}%")
        print(f"Top-5 Accuracy: {top5:.3f}%")
    else:
        top1 = None
        top5 = None
        print("Top-1 Accuracy: not available")
        print("Top-5 Accuracy: not available")
        print(
            "Reason: Oxford-IIIT Pet has 37 labels, "
            f"but this model outputs {len(output_categories)} classes."
        )
    return {
        "name": name,
        "images": total,
        "metrics_compatible": metrics_compatible,
        "target_classes": len(target_categories),
        "output_classes": len(output_categories),
        "top1_accuracy": top1,
        "top5_accuracy": top5,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate EfficientNet V2-L on Oxford-IIIT Pet only."
    )
    parser.add_argument("--models", choices=["both", "default", "finetuned"], default="both")
    parser.add_argument("--checkpoint", default="efficientnetv2l_hf_imagenet1k_finetuned.pth")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--print-predictions", action="store_true")
    parser.add_argument("--result-json", default="result.json")

    parser.add_argument("--pet-root", default=".")
    parser.add_argument("--pet-split", choices=["trainval", "test"], default="test")
    parser.add_argument("--download-pet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = pick_device()
    print("Device:", device)

    weights = EfficientNet_V2_L_Weights.IMAGENET1K_V1
    transform = weights.transforms()
    dataset = load_pet_dataset(args, transform)
    target_categories = dataset.classes
    imagenet_categories = weights.meta["categories"]

    results = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "models": args.models,
        "dataset": "Oxford-IIIT Pet",
        "pet_root": args.pet_root,
        "pet_split": args.pet_split,
        "checkpoint": args.checkpoint,
        "batch_size": args.batch_size,
        "results": [],
    }

    if args.models in {"both", "default"}:
        model, num_classes = build_default_model(weights, device)
        output_categories = target_categories if num_classes == PET_NUM_CLASSES else imagenet_categories
        result = evaluate_dataset(
            "Default model - Oxford-IIIT Pet",
            model,
            dataset,
            target_categories,
            output_categories,
            args,
            device,
        )
        result["model"] = "default"
        results["results"].append(result)
        del model

    if args.models in {"both", "finetuned"}:
        model, num_classes = build_finetuned_model(args.checkpoint, device)
        output_categories = target_categories if num_classes == PET_NUM_CLASSES else imagenet_categories
        result = evaluate_dataset(
            "Fine-tuned model - Oxford-IIIT Pet",
            model,
            dataset,
            target_categories,
            output_categories,
            args,
            device,
        )
        result["model"] = "finetuned"
        results["results"].append(result)
        del model

    result_path = Path(args.result_json)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    print(f"\nSaved results to {result_path}")


if __name__ == "__main__":
    main()
