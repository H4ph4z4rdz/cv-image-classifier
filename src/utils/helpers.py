"""Utility functions for training and evaluation."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(preferred: str = "cuda") -> torch.device:
    """Get the best available device.

    Args:
        preferred: Preferred device ('cuda' or 'cpu').

    Returns:
        torch.device object.
    """
    if preferred == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def save_checkpoint(model, optimizer, epoch, loss, accuracy, path):
    """Save a training checkpoint.

    Args:
        model: The model to save.
        optimizer: The optimizer state to save.
        epoch: Current epoch number.
        loss: Current loss value.
        accuracy: Current accuracy value.
        path: File path to save the checkpoint.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "accuracy": accuracy,
    }, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(model, optimizer, path):
    """Load a training checkpoint.

    Args:
        model: The model to load weights into.
        optimizer: The optimizer to load state into.
        path: File path of the checkpoint.

    Returns:
        Tuple of (epoch, loss, accuracy).
    """
    checkpoint = torch.load(path, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    accuracy = checkpoint["accuracy"]
    print(f"Checkpoint loaded: epoch {epoch}, loss {loss:.4f}, accuracy {accuracy:.2f}%")
    return epoch, loss, accuracy
