"""
AI_stats_lab.py

Autograded lab: Gradient Descent + Linear Regression (Diabetes)

Implemented: Gradient Descent, Analytical solution, Comparison, and Q1 visualization arrays.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

# =========================
# Helpers
# =========================

def add_bias_column(X: np.ndarray) -> np.ndarray:
    """Add a bias (intercept) column of ones to X."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    return np.hstack([np.ones((X.shape[0], 1)), X])

def standardize_train_test(
    X_train: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0, ddof=0)
    sigma = np.where(sigma == 0, 1.0, sigma)
    return (X_train - mu)/sigma, (X_test - mu)/sigma, mu, sigma

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    return float(np.mean((y_true - y_pred)**2))

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res/ss_tot)

@dataclass
class GDResult:
    theta: np.ndarray
    losses: np.ndarray
    thetas: np.ndarray

# =========================
# Q1: Gradient Descent + Visualization
# =========================

import numpy as np
from typing import Optional, Dict
from dataclasses import dataclass

@dataclass
class GDResult:
    theta: np.ndarray
    losses: np.ndarray
    thetas: np.ndarray

# =========================
# Q1: Gradient Descent Implementation
# =========================
def gradient_descent_linreg(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.05,
    epochs: int = 200,
    theta0: Optional[np.ndarray] = None,
) -> GDResult:
    n, d = X.shape
    # Initialize theta
    if theta0 is None:
        theta = np.zeros(d)
    else:
        theta = theta0.copy()
    
    losses = np.zeros(epochs)
    theta_history = np.zeros((epochs, d))
    
    for t in range(epochs):
        # Compute predictions
        y_pred = X @ theta
        # Compute loss
        loss = np.mean((y - y_pred) ** 2)
        losses[t] = loss
        theta_history[t] = theta
        # Gradient
        grad = (-2/n) * (X.T @ (y - y_pred))
        # Update theta
        theta -= lr * grad
    
    return GDResult(theta=theta, losses=losses, thetas=theta_history)

# =========================
# Q1: Visualization Dataset
# =========================
def visualize_gradient_descent(
    lr: float = 0.1,
    epochs: int = 60,
    seed: int = 0
) -> Dict[str, np.ndarray]:
    np.random.seed(seed)
    # Synthetic dataset: y = 1 + 2*x + noise
    n = 50
    x = np.random.uniform(-1, 1, size=(n, 1))
    y = 1 + 2 * x[:, 0] + 0.1 * np.random.randn(n)
    
    # Add bias column
    X = np.hstack([np.ones((n, 1)), x])
    
    # Run gradient descent
    result = gradient_descent_linreg(X, y, lr=lr, epochs=epochs)
    
    return {
        "theta_path": result.thetas,
        "losses": result.losses,
        "X": X,
        "y": y
    }

# =========================
# Optional plotting
# =========================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    data = visualize_gradient_descent(lr=0.1, epochs=60)
    theta_path = data["theta_path"]
    losses = data["losses"]
    
    # Loss vs Epoch
    plt.figure()
    plt.plot(losses, marker='o')
    plt.title("Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.show()
    
    # Theta trajectory
    plt.figure()
    plt.plot(theta_path[:, 0], theta_path[:, 1], marker='o')
    plt.title("Theta Trajectory")
    plt.xlabel("Theta0 (bias)")
    plt.ylabel("Theta1 (weight)")
    plt.grid(True)
    plt.show()

# =========================
# Q2: Diabetes Linear Regression with Gradient Descent
# =========================

def diabetes_linear_gd(
    lr: float = 0.05,
    epochs: int = 2000,
    test_size: float = 0.2,
    seed: int = 0,
) -> Tuple[float, float, float, float, np.ndarray]:
    np.random.seed(seed)
    diabetes = datasets.load_diabetes()
    X, y = diabetes.data, diabetes.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    X_train, X_test, _, _ = standardize_train_test(X_train, X_test)
    X_train = add_bias_column(X_train)
    X_test = add_bias_column(X_test)
    
    gd = gradient_descent_linreg(X_train, y_train, lr=lr, epochs=epochs)
    theta = gd.theta
    
    train_pred = X_train @ theta
    test_pred = X_test @ theta
    return mse(y_train, train_pred), mse(y_test, test_pred), r2_score(y_train, train_pred), r2_score(y_test, test_pred), theta

# =========================
# Q3: Diabetes Linear Regression with Analytical Solution
# =========================

def diabetes_linear_analytical(
    ridge_lambda: float = 1e-8,
    test_size: float = 0.2,
    seed: int = 0,
) -> Tuple[float, float, float, float, np.ndarray]:
    np.random.seed(seed)
    diabetes = datasets.load_diabetes()
    X, y = diabetes.data, diabetes.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    X_train, X_test, _, _ = standardize_train_test(X_train, X_test)
    X_train = add_bias_column(X_train)
    X_test = add_bias_column(X_test)
    
    d = X_train.shape[1]
    theta = np.linalg.inv(X_train.T @ X_train + ridge_lambda*np.eye(d)) @ X_train.T @ y_train
    
    train_pred = X_train @ theta
    test_pred = X_test @ theta
    return mse(y_train, train_pred), mse(y_test, test_pred), r2_score(y_train, train_pred), r2_score(y_test, test_pred), theta

# =========================
# Q4: Compare GD vs Analytical
# =========================

def diabetes_compare_gd_vs_analytical(
    lr: float = 0.05,
    epochs: int = 4000,
    test_size: float = 0.2,
    seed: int = 0,
) -> Dict[str, float]:
    gd_metrics = diabetes_linear_gd(lr=lr, epochs=epochs, test_size=test_size, seed=seed)
    analytical_metrics = diabetes_linear_analytical(test_size=test_size, seed=seed)
    
    theta_gd = gd_metrics[4]
    theta_an = analytical_metrics[4]
    
    # Cosine similarity
    cos_sim = (theta_gd @ theta_an) / (np.linalg.norm(theta_gd)*np.linalg.norm(theta_an))
    
    return {
        "theta_l2_diff": float(np.linalg.norm(theta_gd - theta_an)),
        "train_mse_diff": float(abs(gd_metrics[0] - analytical_metrics[0])),
        "test_mse_diff": float(abs(gd_metrics[1] - analytical_metrics[1])),
        "train_r2_diff": float(abs(gd_metrics[2] - analytical_metrics[2])),
        "test_r2_diff": float(abs(gd_metrics[3] - analytical_metrics[3])),
        "theta_cosine_sim": float(cos_sim)
    }
