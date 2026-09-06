"""
Handwritten Digit Recognition
------------------------------
Trains and compares multiple ML models to recognize handwritten digits (0-9)
using the scikit-learn digits dataset (1,797 8x8 grayscale images).

Run: python3 digit_recognition.py
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
digits = load_digits()
X, y = digits.data, digits.target
print(f"Dataset: {X.shape[0]} images, {X.shape[1]} pixel features each (8x8)")
print(f"Classes: {sorted(set(y))}")

# Visualize a few sample digits
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for ax, image, label in zip(axes.ravel(), digits.images, digits.target):
    ax.imshow(image, cmap="gray")
    ax.set_title(f"Label: {label}")
    ax.axis("off")
plt.tight_layout()
plt.savefig("sample_digits.png", dpi=150)
print("Saved sample_digits.png")

# ---------------------------------------------------------------------------
# 2. Train / test split + scaling
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 3. Train and compare multiple models
# ---------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Support Vector Machine": SVC(kernel="rbf", gamma="scale"),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42),
}

results = {}
best_model_name, best_model, best_acc = None, None, 0.0

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"\n=== {name} (accuracy: {acc:.4f}) ===")
    print(classification_report(y_test, preds))

    if acc > best_acc:
        best_acc, best_model_name, best_model = acc, name, model

print(f"\nBest model: {best_model_name} with accuracy {best_acc:.4f}")

# ---------------------------------------------------------------------------
# 4. Confusion matrix for the best model
# ---------------------------------------------------------------------------
preds = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
disp.plot(cmap="Blues")
plt.title(f"Confusion Matrix - {best_model_name}")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("Saved confusion_matrix.png")

# ---------------------------------------------------------------------------
# 5. Model comparison bar chart
# ---------------------------------------------------------------------------
plt.figure()
plt.bar(results.keys(), results.values(), color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
plt.ylabel("Accuracy")
plt.title("Model Comparison")
plt.ylim(0.8, 1.0)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
print("Saved model_comparison.png")

# ---------------------------------------------------------------------------
# 6. Show some misclassified examples
# ---------------------------------------------------------------------------
misclassified_idx = np.where(preds != y_test)[0]
if len(misclassified_idx) > 0:
    n_show = min(5, len(misclassified_idx))
    fig, axes = plt.subplots(1, n_show, figsize=(2.5 * n_show, 3))
    if n_show == 1:
        axes = [axes]
    for ax, idx in zip(axes, misclassified_idx[:n_show]):
        image = X_test[idx].reshape(8, 8)
        ax.imshow(image, cmap="gray")
        ax.set_title(f"True: {y_test[idx]}\nPred: {preds[idx]}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("misclassified_examples.png", dpi=150)
    print("Saved misclassified_examples.png")
else:
    print("No misclassified examples.")

# ---------------------------------------------------------------------------
# 7. Save the best model + scaler
# ---------------------------------------------------------------------------
joblib.dump(best_model, "digit_model.joblib")
joblib.dump(scaler, "scaler.joblib")
print("Saved digit_model.joblib and scaler.joblib")
