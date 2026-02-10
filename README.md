# CV Image Classifier

A convolutional neural network (CNN) image classifier built with PyTorch. Includes a complete ML pipeline from data loading to training, evaluation, inference, and an interactive web UI for real-time predictions.

## Features

- **SimpleCNN** architecture with 3 convolutional blocks (~1.2M parameters)
- **CIFAR-10** dataset support (10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Data augmentation** (random crop, flip, color jitter) to prevent overfitting
- **Cosine annealing** learning rate schedule with early stopping
- **Interactive web UI** powered by Gradio - upload any image and get predictions
- **GPU accelerated** training with CUDA (tested on RTX 4090)
- **Modular codebase** - easy to swap models, datasets, and hyperparameters

## Project Structure

```
cv-image-classifier/
├── configs/
│   └── default.yaml         # Hyperparameter configuration
├── data/
│   ├── raw/                  # Downloaded dataset (auto-fetched)
│   ├── processed/            # Preprocessed data
│   └── augmented/            # Augmented training data
├── models/
│   ├── saved/                # Best trained models
│   └── checkpoints/          # Periodic checkpoints
├── notebooks/                # Jupyter notebooks for exploration
├── results/
│   ├── plots/                # Confusion matrices, training curves
│   └── logs/                 # Training logs
└── src/
    ├── app.py                # Gradio web UI
    ├── train.py              # Training script
    ├── evaluate.py           # Evaluation with detailed metrics
    ├── predict.py            # CLI single-image inference
    ├── data/
    │   └── dataset.py        # Data loading & transforms
    ├── models/
    │   └── cnn.py            # CNN architecture
    └── utils/
        └── helpers.py        # Seed, device, checkpoint utilities
```

## Setup

```bash
# Clone the repo
git clone git@github.com:H4ph4z4rdz/cv-image-classifier.git
cd cv-image-classifier

# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# Install PyTorch with CUDA (for GPU support)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install remaining dependencies
pip install -r requirements.txt
```

## Usage

### Train the Model

```bash
python src/train.py --config configs/default.yaml
```

Trains a SimpleCNN on CIFAR-10 for 25 epochs with data augmentation. The best model is automatically saved to `models/saved/best_model.pth`.

### Launch the Web UI

```bash
python src/app.py
```

Opens an interactive web interface at **http://localhost:7860** where you can:
- Upload any image (drag & drop or click to browse)
- See real-time predictions with confidence percentages
- The model auto-classifies on upload

### Evaluate on Test Set

```bash
python src/evaluate.py --model models/saved/best_model.pth
```

Generates a full classification report and confusion matrix.

### CLI Prediction

```bash
python src/predict.py --image path/to/image.jpg --model models/saved/best_model.pth
```

## Model Architecture

```
Input (3x32x32) → Conv Block 1 (32 filters) → Conv Block 2 (64 filters)
→ Conv Block 3 (128 filters) → FC(2048→512) → FC(512→10) → Output
```

Each conv block: `Conv2d → BatchNorm → ReLU → Conv2d → BatchNorm → ReLU → MaxPool → Dropout`

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | **83.89%** |
| Parameters | ~1.2M |
| Training Time | ~2 min (RTX 4090) |
| Dataset | CIFAR-10 (60K images) |

## Tech Stack

- **PyTorch** - Deep learning framework
- **torchvision** - Datasets, transforms, pretrained models
- **Gradio** - Interactive web UI for predictions
- **Matplotlib / Seaborn** - Visualization
- **scikit-learn** - Evaluation metrics
- **CUDA** - GPU acceleration (RTX 4090)

## Configuration

Edit `configs/default.yaml` to tweak hyperparameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 64 | Images per training step |
| `epochs` | 25 | Training iterations over full dataset |
| `learning_rate` | 0.001 | Step size for weight updates |
| `augmentation` | true | Enable data augmentation |
| `early_stopping_patience` | 5 | Stop if no improvement for N epochs |

## License

MIT
