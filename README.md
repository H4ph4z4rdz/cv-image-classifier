# CV Image Classifier

A convolutional neural network (CNN) image classifier built with PyTorch. This project implements a complete ML pipeline from data loading to training, evaluation, and inference.

## Project Structure

```
cv-image-classifier/
├── configs/             # Training configuration files
├── data/
│   ├── raw/             # Original dataset
│   ├── processed/       # Preprocessed data
│   └── augmented/       # Augmented training data
├── models/
│   ├── saved/           # Final trained models
│   └── checkpoints/     # Training checkpoints
├── notebooks/           # Jupyter notebooks for exploration
├── results/
│   ├── plots/           # Training curves, confusion matrices
│   └── logs/            # Training logs
└── src/
    ├── data/            # Data loading and preprocessing
    ├── models/          # Model architectures
    └── utils/           # Helper functions
```

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Train the model
python src/train.py --config configs/default.yaml

# Evaluate on test set
python src/evaluate.py --model models/saved/best_model.pth

# Run inference on a single image
python src/predict.py --image path/to/image.jpg --model models/saved/best_model.pth
```

## Tech Stack

- **PyTorch** - Deep learning framework
- **torchvision** - Datasets, transforms, pretrained models
- **Matplotlib / Seaborn** - Visualization
- **CUDA** - GPU acceleration (RTX 4090)

## License

MIT
