# Handwritten Digit Recognition

A Computer Vision project that recognizes handwritten digits (0–9) using the classic scikit-learn `digits` dataset (1,797 images, 8×8 grayscale) and compares four machine learning classifiers.

## Overview

1. **Data** — the built-in scikit-learn `digits` dataset (no external download needed).
2. **Preprocessing** — feature scaling with `StandardScaler`.
3. **Models compared** — Logistic Regression, Support Vector Machine (RBF kernel), Random Forest, and a Neural Network (MLP).
4. **Evaluation** — accuracy, per-class precision/recall/F1, confusion matrix, and visual inspection of misclassified digits.
5. **Model saving** — best-performing model persisted with `joblib`.

## Project Structure

```
project2-digit-recognition/
├── digit_recognition.py         # standalone script version
├── Digit_Recognition.ipynb      # notebook walkthrough with visualizations
├── requirements.txt
└── README.md
```

## How to Run

```bash
pip install -r requirements.txt
python3 digit_recognition.py
```

Or open `Digit_Recognition.ipynb` in Jupyter for the full walkthrough with sample digit images and charts.

## Results

The Support Vector Machine achieved the best accuracy (~97.5%) on the held-out test set, with the Neural Network close behind. Digits like 8 and 9 are the most commonly confused, which matches how humans often misread hastily-written digits too.

## Tech Stack

- Python, scikit-learn, matplotlib, numpy, joblib

## Possible Extensions

- Scale up to the full MNIST dataset (28×28, 70,000 images) with a CNN (TensorFlow/PyTorch)
- Add data augmentation (rotation, noise) to test robustness
- Build a simple web app where users draw a digit and get a live prediction
