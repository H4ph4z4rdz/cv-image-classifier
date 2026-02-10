"""Gradio web UI for the image classifier.

Launch with:
    python src/app.py

Then open http://localhost:7860 in your browser.
Upload any image and the model will predict what it is!
"""

import os
import sys

import torch
import gradio as gr
import numpy as np
from PIL import Image
from torchvision import transforms

# Add project root to path so imports work
sys.path.insert(0, os.path.dirname(__file__))
from models import SimpleCNN
from utils import get_device


# CIFAR-10 class names with emoji for fun
CLASSES = {
    "airplane": "✈️ Airplane",
    "automobile": "🚗 Automobile",
    "bird": "🐦 Bird",
    "cat": "🐱 Cat",
    "deer": "🦌 Deer",
    "dog": "🐕 Dog",
    "frog": "🐸 Frog",
    "horse": "🐴 Horse",
    "ship": "🚢 Ship",
    "truck": "🚛 Truck",
}

CLASS_NAMES = list(CLASSES.keys())
CLASS_LABELS = list(CLASSES.values())

# Preprocessing transform (same as training, no augmentation)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])


def load_model(model_path: str = "models/saved/best_model.pth"):
    """Load the trained model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleCNN(num_classes=10).to(device)

    if not os.path.exists(model_path):
        print(f"WARNING: Model not found at {model_path}")
        print("Please train the model first: python src/train.py")
        return None, device

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    accuracy = checkpoint.get("accuracy", "unknown")
    epoch = checkpoint.get("epoch", "unknown")
    print(f"Model loaded from {model_path}")
    print(f"  Trained for {epoch} epochs, test accuracy: {accuracy}%")

    return model, device


def predict(image):
    """Run prediction on an uploaded image.

    Args:
        image: PIL Image from Gradio upload.

    Returns:
        Dictionary of {class_label: confidence} for Gradio Label output.
    """
    if image is None:
        return {}

    if model is None:
        return {"Error: Model not loaded": 1.0}

    # Preprocess
    img = Image.fromarray(image).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    # Build results dictionary
    results = {}
    for i, (name, label) in enumerate(zip(CLASS_NAMES, CLASS_LABELS)):
        results[label] = float(probabilities[i])

    return results


# Load model at startup
model, device = load_model()

# Build the Gradio interface
with gr.Blocks(
    title="CV Image Classifier",
    theme=gr.themes.Soft(primary_hue="blue"),
) as demo:

    gr.Markdown(
        """
        # 🧠 CV Image Classifier
        ### A CNN trained on CIFAR-10 — upload an image and see what the model thinks it is!

        The model can recognize: ✈️ Airplanes, 🚗 Automobiles, 🐦 Birds, 🐱 Cats,
        🦌 Deer, 🐕 Dogs, 🐸 Frogs, 🐴 Horses, 🚢 Ships, and 🚛 Trucks.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="Upload an Image",
                type="numpy",
                height=300,
            )
            predict_btn = gr.Button("🔍 Classify Image", variant="primary", size="lg")

            gr.Markdown(
                """
                **Tips:**
                - Works best with clear photos of single objects
                - The model was trained on tiny 32×32 images, so it resizes your upload
                - Try different angles and lighting conditions!
                """
            )

        with gr.Column(scale=1):
            label_output = gr.Label(
                label="Predictions",
                num_top_classes=5,
            )

            gr.Markdown(
                """
                **How it works:**
                1. Your image is resized to 32×32 pixels
                2. Pixel values are normalized
                3. The CNN processes it through 3 conv blocks
                4. The classifier outputs probabilities for each class
                """
            )

    # Wire up the button
    predict_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=label_output,
    )

    # Also predict on image upload
    image_input.change(
        fn=predict,
        inputs=image_input,
        outputs=label_output,
    )

    gr.Markdown(
        """
        ---
        *Built with PyTorch & Gradio | Model: SimpleCNN | Dataset: CIFAR-10 |
        GPU: NVIDIA RTX 4090*
        """
    )


if __name__ == "__main__":
    print("\nLaunching Image Classifier UI...")
    print("   Open http://localhost:7860 in your browser\n")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
