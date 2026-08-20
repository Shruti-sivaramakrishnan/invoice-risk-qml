"""
train_and_compare.py
Trains classical baselines (Logistic Regression, RBF-SVM) and a quantum
kernel SVM (PennyLane, 5-qubit angle embedding with entangling layers),
then compares them on held-out test data.
"""

import os
import time
import numpy as np
import pandas as pd
import pennylane as qml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")
FEATURES_PATH = os.path.join(PROCESSED_DIR, "features.csv")
METRICS_OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "metrics_comparison.csv")
os.makedirs(FIG_DIR, exist_ok=True)

FEATURE_COLS = [
    "amount_zscore_norm", "threshold_proximity",
    "duplicate_score", "weekend_flag", "days_to_due_norm",
]
N_QUBITS = 5

# ---------------------------------------------------------------------------
# Quantum kernel (ZZFeatureMap-style angle embedding + entangling layer)
# ---------------------------------------------------------------------------
dev = qml.device("default.qubit", wires=N_QUBITS)


def feature_map(x):
    for i in range(N_QUBITS):
        qml.Hadamard(wires=i)
        qml.RZ(2 * x[i], wires=i)
    for i in range(N_QUBITS - 1):
        qml.CNOT(wires=[i, i + 1])
        qml.RZ(2 * (np.pi - x[i]) * (np.pi - x[i + 1]), wires=i + 1)
        qml.CNOT(wires=[i, i + 1])


@qml.qnode(dev)
def kernel_circuit(x1, x2):
    feature_map(x1)
    qml.adjoint(feature_map)(x2)
    return qml.probs(wires=range(N_QUBITS))


def quantum_kernel(x1, x2):
    return kernel_circuit(x1, x2)[0]  # probability of measuring |00000>


def build_kernel_matrix(X1, X2):
    n1, n2 = len(X1), len(X2)
    K = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            K[i, j] = quantum_kernel(X1[i], X2[j])
    return K


def main():
    df = pd.read_csv(FEATURES_PATH)
    X = df[FEATURE_COLS].values
    y = df["label"].values

    # Scale features to [0, pi] for angle embedding
    X_angles = X * np.pi

    X_train, X_test, y_train, y_test = train_test_split(
        X_angles, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}  Test: {len(X_test)}  "
          f"(train risky: {y_train.sum()}, test risky: {y_test.sum()})")

    results = {}

    # --- Logistic Regression ---
    t0 = time.time()
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    results["Logistic Regression"] = {
        "accuracy": accuracy_score(y_test, lr_pred),
        "precision": precision_score(y_test, lr_pred, zero_division=0),
        "recall": recall_score(y_test, lr_pred, zero_division=0),
        "f1": f1_score(y_test, lr_pred, zero_division=0),
        "train_time_sec": time.time() - t0,
        "confusion_matrix": confusion_matrix(y_test, lr_pred),
    }

    # --- RBF-SVM (C=10, per prior validation sweep) ---
    t0 = time.time()
    svm = SVC(kernel="rbf", C=10)
    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_test)
    results["RBF-SVM"] = {
        "accuracy": accuracy_score(y_test, svm_pred),
        "precision": precision_score(y_test, svm_pred, zero_division=0),
        "recall": recall_score(y_test, svm_pred, zero_division=0),
        "f1": f1_score(y_test, svm_pred, zero_division=0),
        "train_time_sec": time.time() - t0,
        "confusion_matrix": confusion_matrix(y_test, svm_pred),
    }

    # --- Quantum Kernel SVM ---
    t0 = time.time()
    K_train = build_kernel_matrix(X_train, X_train)
    K_test = build_kernel_matrix(X_test, X_train)
    qsvm = SVC(kernel="precomputed", C=10)
    qsvm.fit(K_train, y_train)
    q_pred = qsvm.predict(K_test)
    results["Quantum Kernel SVM"] = {
        "accuracy": accuracy_score(y_test, q_pred),
        "precision": precision_score(y_test, q_pred, zero_division=0),
        "recall": recall_score(y_test, q_pred, zero_division=0),
        "f1": f1_score(y_test, q_pred, zero_division=0),
        "train_time_sec": time.time() - t0,
        "confusion_matrix": confusion_matrix(y_test, q_pred),
    }

    # --- Save metrics table ---
    rows = []
    for model_name, m in results.items():
        rows.append({
            "model": model_name,
            "accuracy": round(m["accuracy"], 3),
            "precision": round(m["precision"], 3),
            "recall": round(m["recall"], 3),
            "f1": round(m["f1"], 3),
            "train_time_sec": round(m["train_time_sec"], 3),
        })
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(METRICS_OUT, index=False)
    print("\n" + metrics_df.to_string(index=False))

    # --- Bar chart of F1 + runtime ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(metrics_df["model"], metrics_df["f1"], color=["#4C72B0", "#55A868", "#C44E52"])
    axes[0].set_title("F1 Score by Model")
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(metrics_df["model"], metrics_df["train_time_sec"], color=["#4C72B0", "#55A868", "#C44E52"])
    axes[1].set_title("Training Time (seconds)")
    axes[1].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "metrics_comparison.png"), dpi=150)
    plt.close()

    # --- Confusion matrices ---
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, (name, m) in zip(axes, results.items()):
        cm = m["confusion_matrix"]
        ax.imshow(cm, cmap="Blues")
        ax.set_title(name, fontsize=9)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "confusion_matrices.png"), dpi=150)
    plt.close()

    print(f"\nFigures saved to {FIG_DIR}")
    print(f"Metrics table saved to {METRICS_OUT}")


if __name__ == "__main__":
    main()
