# CIFAR-10 Image Classification with a CNN (PyTorch)

A convolutional neural network built from scratch in PyTorch to classify the CIFAR-10 dataset — including model training, evaluation, and a hyperparameter tuning study.

## Overview

This project implements and trains a CNN to classify CIFAR-10 images into 10 categories. Beyond training a single model, it explores how performance changes across a grid of hyperparameters (filter counts, learning rate, batch size), and analyzes the model's errors through a confusion matrix and misclassified-example inspection.

Built as part of a Machine Learning course at Hochschule Düsseldorf (Team 9) — the model design, training, evaluation, and hyperparameter tuning in this repo were my individual contribution.

## What it does

- Trains a custom CNN (`SimpleCNN`) on CIFAR-10 from scratch
- Evaluates the trained model on the held-out test set
- Runs a hyperparameter grid search to find the best-performing configuration
- Visualizes training/validation loss & accuracy curves, a confusion matrix, and misclassified examples

## Model Architecture

`SimpleCNN`:
- `Conv2d(3→32, 3x3)` → ReLU → `MaxPool(2x2)`
- `Conv2d(32→64, 3x3)` → ReLU → `MaxPool(2x2)`
- Flatten → `Linear(4096→128)` → ReLU → `Linear(128→10)`

## Results

| Metric | Value |
|---|---|
| Test Accuracy | 69.58% |
| Test Loss | 1.4553 |

Training showed a classic overfitting pattern: training loss kept decreasing and training accuracy kept climbing, while validation loss started rising (and validation accuracy dropping) after epoch 5 — a sign the model began memorizing the training data rather than generalizing.

**Common misclassifications** (from the confusion matrix): cat ↔ dog, automobile ↔ truck, deer ↔ horse, airplane ↔ bird — mostly visually or semantically similar classes. Expected given CIFAR-10's small 32×32 images, similar colors/shapes across some classes, and no data augmentation.

### Hyperparameter tuning

Grid search over filter counts (32/64, 32/128, 64/64, 64/128), learning rate (0.001, 0.01), and batch size (64, 128), on a 10% data subset for speed:

| Configuration | Result |
|---|---|
| Best: `c1=64, c2=64, lr=0.001, batch=128` | **56.8%** validation accuracy |
| High learning rate (0.01) | 8–30% (unstable training) |
| Standard learning rate (0.001) | 53–56% (stable, best results) |

Takeaway: a learning rate that's too high destabilizes training regardless of filter count; more filters and a bigger batch size don't automatically help.

## Project Structure

```
CNN/
├── cnn_cifar10.py            # Model definition + training loop
├── train_loader.py           # CIFAR-10 loading, normalization, train/val/test split
├── evaluate_cnn_cifar10.py   # Test-set evaluation, confusion matrix, misclassified examples
├── hyperparameter_tuning.py  # Grid search over filters/lr/batch size
└── cnn_cifar10_model.pth     # Trained model weights
```

## How to Run

```bash
pip install torch torchvision matplotlib scikit-learn pandas numpy
```

Download the CIFAR-10 dataset (or set `download=True` in `train_loader.py` if it isn't present locally), then:

```bash
python cnn_cifar10.py            # train the model
python evaluate_cnn_cifar10.py   # evaluate on the test set
python hyperparameter_tuning.py  # run the grid search
```

## Possible Improvements

- Data augmentation and/or dropout to reduce overfitting
- Deeper architecture / batch normalization
- Learning rate scheduling

---

Part of a Machine Learning course project at Hochschule Düsseldorf. See also: **ML Pipeline** — the CI/CD automation of this model *(link once that repo is live)*.
