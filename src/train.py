"""Training script for the image classifier."""

import argparse
import os
import time

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data import get_dataloaders
from models import SimpleCNN
from utils import set_seed, get_device, save_checkpoint


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch.

    Returns:
        Tuple of (average_loss, accuracy_percentage).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct/total:.1f}%")

    avg_loss = running_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, test_loader, criterion, device):
    """Evaluate the model on test data.

    Returns:
        Tuple of (average_loss, accuracy_percentage).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(test_loader, desc="Evaluating", leave=False):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / len(test_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Train the image classifier")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config file")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Setup
    set_seed(config.get("seed", 42))
    device = get_device(config.get("device", "cuda"))

    # Data
    print("\nLoading dataset...")
    train_loader, test_loader, class_names = get_dataloaders(
        dataset_name=config["data"]["dataset"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        image_size=config["data"]["image_size"],
        augment=config["data"]["augmentation"],
    )
    print(f"Classes: {class_names}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Model
    model = SimpleCNN(num_classes=config["model"]["num_classes"]).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {config['model']['name']} ({total_params:,} parameters)")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["epochs"]
    )

    # Training loop
    best_accuracy = 0.0
    patience_counter = 0
    patience = config["training"].get("early_stopping_patience", 5)

    print(f"\nStarting training for {config['training']['epochs']} epochs...\n")

    for epoch in range(1, config["training"]["epochs"] + 1):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        scheduler.step()
        elapsed = time.time() - start_time

        print(
            f"Epoch [{epoch}/{config['training']['epochs']}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | "
            f"Time: {elapsed:.1f}s"
        )

        # Save best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            patience_counter = 0
            save_checkpoint(
                model, optimizer, epoch, test_loss, test_acc,
                os.path.join(config["save_dir"], "best_model.pth"),
            )
        else:
            patience_counter += 1

        # Save periodic checkpoint
        if epoch % 5 == 0:
            save_checkpoint(
                model, optimizer, epoch, test_loss, test_acc,
                os.path.join(config["checkpoint_dir"], f"checkpoint_epoch_{epoch}.pth"),
            )

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    print(f"\nTraining complete! Best test accuracy: {best_accuracy:.2f}%")


if __name__ == "__main__":
    main()
