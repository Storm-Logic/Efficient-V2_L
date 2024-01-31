import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights
from tqdm import tqdm


def main():

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print("Device:", device)

    # official transforms
    weights = EfficientNet_V2_L_Weights.IMAGENET1K_V1
    transform = weights.transforms()

    # validation dataset
    dataset = ImageNet(
        root="./imagenet",
        split="val",
        transform=transform
    )

    prediction_class_names = weights.meta["categories"]
    true_class_names = [names[0] for names in dataset.classes]

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # rebuild model
    model = efficientnet_v2_l(weights=None)

    # ImageNet-1K = 1000 classes
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1000)

    # load trained weights
    checkpoint = torch.load(
        "efficientnetv2l_val_finetuned.pth",
        map_location=device
    )

    # if saved using model.state_dict()
    model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    top1_correct = 0
    top5_correct = 0
    total = 0
    predictions = []

    with torch.no_grad():

        for images, labels in tqdm(loader):

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, pred = outputs.topk(5, dim=1)

            total += labels.size(0)

            top1_correct += (pred[:, 0] == labels).sum().item()

            top5_correct += pred.eq(
                labels.view(-1, 1)
            ).sum().item()

            batch_start = total - labels.size(0)

            for offset in range(labels.size(0)):
                image_path = dataset.samples[batch_start + offset][0]
                true_index = labels[offset].item()
                top5_indices = pred[offset].cpu().tolist()
                top5_names = [prediction_class_names[index] for index in top5_indices]

                predictions.append({
                    "image_path": image_path,
                    "true_class": true_class_names[true_index],
                    "predicted_class": top5_names[0],
                    "top5_classes": "; ".join(top5_names),
                })

                print(
                    f"{image_path}: predicted={top5_names[0]} | true={true_class_names[true_index]}"
                )


    top1 = 100 * top1_correct / total
    top5 = 100 * top5_correct / total

    print(f"Top-1 Accuracy: {top1:.3f}%")
    print(f"Top-5 Accuracy: {top5:.3f}%")


if __name__ == "__main__":
    main()