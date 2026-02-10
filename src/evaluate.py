"""Evaluation script with detailed metrics and visualizations."""

import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from data import get_dataloaders
from models import SimpleCNN
from utils import get_device


@torch.no_grad()
def get_predictions(model, data_loader, device):
    """Get all predictions and labels from the test set.

    Returns:
        Tuple of (all_predictions, all_labels, all_probabilities).
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in tqdm(data_loader, desc="Getting predictions"):
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, preds = outputs.max(1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def plot_confusion_matrix(labels, predictions, class_names, save_path=None):
    """Plot and optionally save a confusion matrix."""
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Confusion matrix saved to {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained model")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to saved model")
    parser.add_argument("--num-classes", type=int, default=10)
    args = parser.parse_args()

    device = get_device()

    # Load data
    _, test_loader, class_names = get_dataloaders()

    # Load model
    model = SimpleCNN(num_classes=args.num_classes).to(device)
    checkpoint = torch.load(args.model, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded model from {args.model}")
    print(f"  Epoch: {checkpoint['epoch']}, Accuracy: {checkpoint['accuracy']:.2f}%")

    # Get predictions
    predictions, labels, probabilities = get_predictions(model, test_loader, device)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(labels, predictions, target_names=class_names))

    # Confusion matrix
    plot_confusion_matrix(labels, predictions, class_names, "results/plots/confusion_matrix.png")


if __name__ == "__main__":
    main()
