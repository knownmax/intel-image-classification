"""
evaluate.py — Load best checkpoint and evaluate on the test set.

Outputs:
  • Classification report (stdout)
  • results/confusion_matrix.png
  • results/gradcam_grid.png  (8 sample images)
  • Top-1 and Top-5 accuracy (stdout)

Usage:
    python src/evaluate.py --config configs/train_config.yaml
"""

import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from dataset import get_dataloaders, _build_transforms
from model import get_model


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def top_k_accuracy(outputs: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
    """Compute top-k accuracy from logits."""
    with torch.no_grad():
        _, top_k_preds = outputs.topk(k, dim=1, largest=True, sorted=True)
        correct = top_k_preds.eq(targets.view(-1, 1).expand_as(top_k_preds))
        return correct.any(dim=1).float().mean().item()


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(model, loader, device, classes):
    """
    Run inference on the full loader.

    Returns:
        all_preds, all_targets, all_outputs  (as numpy arrays / tensors)
    """
    model.eval()
    all_preds, all_targets, all_outputs = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="evaluating"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            all_preds.append(logits.argmax(dim=1).cpu())
            all_targets.append(labels.cpu())
            all_outputs.append(logits.cpu())

    all_preds   = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    all_outputs = torch.cat(all_outputs)
    return all_preds, all_targets, all_outputs


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def save_confusion_matrix(targets, preds, classes, save_path: Path) -> None:
    """Save a seaborn confusion matrix heatmap."""
    cm = confusion_matrix(targets, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=classes, yticklabels=classes, ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (normalised)")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] Confusion matrix saved → {save_path}")


# ---------------------------------------------------------------------------
# Grad-CAM grid
# ---------------------------------------------------------------------------

def save_gradcam_grid(
    model, test_dataset, classes, device, save_path: Path, n_samples: int = 8
) -> None:
    """
    Generate Grad-CAM visualisations for n_samples test images and save a grid.

    Uses pytorch-grad-cam targeting the last Conv block of EfficientNetV2-S.
    """
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image

    # Target the last convolutional block of the backbone
    target_layers = [model.backbone.blocks[-1][-1]]

    cam = GradCAM(model=model, target_layers=target_layers)

    # Pick n_samples random indices
    indices = random.sample(range(len(test_dataset)), n_samples)

    raw_transform = _build_transforms("test")   # includes Normalize
    inv_mean = [-m / s for m, s in zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    inv_std  = [1.0 / s for s in [0.229, 0.224, 0.225]]

    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 3, 6))

    for col, idx in enumerate(indices):
        img_tensor, label = test_dataset[idx]
        input_tensor = img_tensor.unsqueeze(0).to(device)

        grayscale_cam = cam(input_tensor=input_tensor)[0]

        # De-normalise for display
        img_np = img_tensor.clone()
        for t, m, s in zip(img_np, inv_mean, inv_std):
            t.mul_(s).add_(m)
        img_np = img_np.permute(1, 2, 0).clamp(0, 1).numpy().astype(np.float32)

        cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

        axes[0, col].imshow(img_np)
        axes[0, col].set_title(classes[label], fontsize=8)
        axes[0, col].axis("off")

        axes[1, col].imshow(cam_image)
        axes[1, col].set_title("Grad-CAM", fontsize=8)
        axes[1, col].axis("off")

    plt.suptitle("Original (top) vs Grad-CAM (bottom)")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] Grad-CAM grid saved → {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained classifier")
    parser.add_argument("--config", default="configs/train_config.yaml")
    args = parser.parse_args()

    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"[evaluate] device = {device}")

    # Load model
    model = get_model(num_classes=cfg.num_classes).to(device)
    ckpt_path = Path(cfg.checkpoint_dir) / cfg.checkpoint_name
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[evaluate] Loaded checkpoint: {ckpt_path}  (val_acc={checkpoint['val_acc']:.4f})")

    # Data
    _, _, test_loader = get_dataloaders(cfg.data_dir, cfg.batch_size, cfg.num_workers)
    from torchvision import datasets
    test_dataset = datasets.ImageFolder(
        root=str(Path(cfg.data_dir) / "seg_test" / "seg_test"),
        transform=_build_transforms("test"),
    )
    classes = cfg.classes

    # Inference
    preds, targets, outputs = evaluate(model, test_loader, device, classes)

    # Top-1 / Top-5
    top1 = (preds == targets).mean()
    k = min(5, cfg.num_classes)
    top5 = top_k_accuracy(outputs, torch.tensor(targets), k=k)
    print(f"\n[evaluate] Top-1 Accuracy : {top1:.4f} ({top1*100:.2f}%)")
    print(f"[evaluate] Top-{k} Accuracy : {top5:.4f} ({top5*100:.2f}%)")

    # Classification report
    print("\n" + classification_report(targets, preds, target_names=classes))

    # Confusion matrix
    results_dir = Path(cfg.checkpoint_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    save_confusion_matrix(targets, preds, classes, results_dir / "confusion_matrix.png")

    # Grad-CAM
    save_gradcam_grid(model, test_dataset, classes, device,
                      results_dir / "gradcam_grid.png", n_samples=8)


if __name__ == "__main__":
    main()
