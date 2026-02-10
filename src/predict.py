"""Inference script for single image prediction."""

import argparse

import torch
from torchvision import transforms
from PIL import Image

from models import SimpleCNN
from utils import get_device


# CIFAR-10 class names
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def predict_image(image_path: str, model, device, class_names=None):
    """Predict the class of a single image.

    Args:
        image_path: Path to the input image.
        model: Trained model.
        device: Device to run inference on.
        class_names: List of class names.

    Returns:
        Tuple of (predicted_class, confidence, all_probabilities).
    """
    if class_names is None:
        class_names = CIFAR10_CLASSES

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = probabilities.max(1)

    predicted_class = class_names[predicted.item()]
    confidence_pct = confidence.item() * 100

    return predicted_class, confidence_pct, probabilities.cpu().numpy()[0]


def main():
    parser = argparse.ArgumentParser(description="Predict image class")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to saved model")
    parser.add_argument("--num-classes", type=int, default=10)
    args = parser.parse_args()

    device = get_device()

    # Load model
    model = SimpleCNN(num_classes=args.num_classes).to(device)
    checkpoint = torch.load(args.model, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Predict
    predicted_class, confidence, probs = predict_image(
        args.image, model, device
    )

    print(f"\nPrediction: {predicted_class}")
    print(f"Confidence: {confidence:.1f}%")
    print("\nAll probabilities:")
    for name, prob in sorted(zip(CIFAR10_CLASSES, probs), key=lambda x: -x[1]):
        bar = "█" * int(prob * 40)
        print(f"  {name:>12}: {prob*100:5.1f}% {bar}")


if __name__ == "__main__":
    main()
