# Waste Classification with CNN

A convolutional neural network (CNN) built with TensorFlow/Keras to classify waste images into **Organic (O)** and **Recyclable (R)** categories.

## Dataset

[Waste Classification Data](https://www.kaggle.com/datasets/techsash/waste-classification-data) from Kaggle (`techsash/waste-classification-data`), downloaded automatically via `kagglehub`.

```
DATASET/
├── TRAIN/
│   ├── O/   (Organic)
│   └── R/   (Recyclable)
└── TEST/
    ├── O/
    └── R/
```

## Model Architecture

A custom sequential CNN trained from scratch on 128×128 RGB images:

| Layer | Output Shape | Details |
|-------|-------------|---------|
| Conv2D | 128×128×32 | 3×3, ReLU, same padding |
| BatchNorm + MaxPool | 64×64×32 | |
| Conv2D | 64×64×64 | 3×3, ReLU, same padding |
| BatchNorm + MaxPool | 32×32×64 | |
| Conv2D | 32×32×128 | 3×3, ReLU, same padding |
| BatchNorm + MaxPool | 16×16×128 | |
| Conv2D | 16×16×256 | 3×3, ReLU, same padding |
| BatchNorm + MaxPool | 8×8×256 | |
| GlobalAveragePooling2D | 256 | |
| Dense + Dropout(0.5) | 256 | ReLU |
| Dense (softmax) | num_classes | Output |

## Training

- **Optimizer:** Adam (lr=0.001)
- **Loss:** Categorical Cross-Entropy
- **Epochs:** up to 30 with early stopping (patience=8)
- **Batch size:** 16
- **Input size:** 128×128

### Data Augmentation (train set)

Rotation ±25°, width/height shift ±15%, shear 0.1, zoom ±20%, horizontal flip.

### Callbacks

- `EarlyStopping` — monitors `val_loss`, restores best weights
- `ModelCheckpoint` — saves best model to `best_waste_cnn.h5`

## Usage

1. Install dependencies:
   ```bash
   pip install tensorflow kagglehub matplotlib pillow
   ```

2. Open and run `Waste_Classification.ipynb` end-to-end. The notebook will:
   - Download the dataset from Kaggle automatically
   - Train the CNN
   - Save `best_waste_cnn.h5` (best checkpoint) and `waste_cnn_final.h5` (final weights)

## Files

| File | Description |
|------|-------------|
| `Waste_Classification.ipynb` | Training notebook |
| `waste_cnn_final.h5` | Saved final model weights |

## Requirements

- Python 3.10+
- TensorFlow 2.x
- kagglehub
- matplotlib, Pillow

GPU recommended for reasonable training time.