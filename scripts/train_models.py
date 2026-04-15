"""
Project: LiDAR Fingerprint Localization using ConvMLP pose regression
File: train_models.py
 

Author: Harsha Keladi Ganapathi
Affiliation: Robotics Lab,
             University of New Haven, CT

License: MIT License
"""

#!/usr/bin/env python3
"""
Description:
- Loads dataset from (dataset/processed) file directory (X.npy, y.npy)
- Evaluates sklearn baselines
- Trains three PyTorch models: MLP, 1D-CNN, CNN+MLP (ConvMLP) hybrid (with Dropout)
- Performs MC-dropout inference (configurable T)
- Produces plots, prediction CSVs, JSON logs
- Optionally compute SHAP explanations (if shap installed)
"""

import os
import json
import time
from datetime import datetime
import pickle
import math

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity

# sklearn baselines
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

# plotting
import matplotlib.pyplot as plt

# --- PyTorch ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Paths
DATA_DIR = "dataset/processed/"
MODEL_DIR = "models/"
PLOTS_DIR = os.path.join(MODEL_DIR, "plots")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# -------------------------
# Utilities / Metrics
# -------------------------
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def cos_sim_mean(y_true, y_pred):
    # compute pairwise cosine, then average diagonal
    sim = cosine_similarity(y_true, y_pred)
    return float(np.mean(np.diag(sim)))

# -------------------------
# Load data
# -------------------------
def load_data():
    X = np.load(os.path.join(DATA_DIR, "X.npy"))
    y = np.load(os.path.join(DATA_DIR, "y.npy"))
    # if scaler exists, load for info
    scaler = None
    try:
        scaler = pickle.load(open(os.path.join(DATA_DIR, "scaler.pkl"), "rb"))
    except Exception:
        pass
    return X, y, scaler

# -------------------------
# PyTorch Dataset wrapper
# -------------------------
class LidarDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -------------------------
# PyTorch Models
# -------------------------
class PyMLP(nn.Module):
    def __init__(self, input_dim, dropout_p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    def forward(self, x):
        return self.net(x)

class Conv1DNet(nn.Module):
    def __init__(self, input_dim, dropout_p=0.3):
        super().__init__()
        # input shape (B, input_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout_p),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_p),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_p)
        )
        reduced = input_dim // 4  # because two poolings
        self.head = nn.Sequential(
            nn.Linear(64 * reduced, 256),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, input_dim)
        x = self.conv(x)
        x = x.flatten(1)
        return self.head(x)

class ConvMLPNet(nn.Module):
    def __init__(self, input_dim, dropout_p=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_p),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_p),
        )
        reduced = input_dim // 4
        self.mlp = nn.Sequential(
            nn.Linear(64 * reduced, 512),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.flatten(1)
        return self.mlp(x)

# -------------------------
# Training loop for PyTorch models
# -------------------------
def train_torch_model(model, train_ds, val_ds,
                      epochs=70, batch_size=64, lr=1e-3,
                      weight_decay=1e-5, verbose=True):
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    history = {"train_loss": [], "val_loss": []}

    for ep in range(1, epochs+1):
        model.train()
        running = 0.0
        for Xb, yb in train_loader:
            Xb = Xb.to(DEVICE); yb = yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(Xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * Xb.shape[0]
        train_loss = running / len(train_ds)
        # val
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for Xv, yv in val_loader:
                Xv = Xv.to(DEVICE); yv = yv.to(DEVICE)
                pv = model(Xv)
                l = loss_fn(pv, yv)
                val_running += l.item() * Xv.shape[0]
        val_loss = val_running / len(val_ds)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if verbose and (ep % 5 == 0 or ep == 1 or ep == epochs):
            print(f"Epoch {ep}/{epochs} — train_loss: {train_loss:.5f}, val_loss: {val_loss:.5f}")
    return model, history

# -------------------------
# MC Dropout inference (T forward passes)
# -------------------------
def mc_dropout_predict(model, X_np, T=50, batch_size=256):
    """
    Returns: mean_preds (N,2), std_preds (N,2)
    model should have dropout layers; we set model to train() for stochastic forward passes
    """
    model.to(DEVICE)
    model.train()  # enable dropout
    ds = LidarDataset(X_np, np.zeros((len(X_np),2), dtype=np.float32))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    all_preds = []
    with torch.no_grad():
        for t in range(T):
            preds_t = []
            for Xb, _ in loader:
                Xb = Xb.to(DEVICE)
                out = model(Xb).cpu().numpy()
                preds_t.append(out)
            preds_t = np.vstack(preds_t)
            all_preds.append(preds_t)
    all_preds = np.stack(all_preds, axis=0)  # (T, N, 2)
    mean = np.mean(all_preds, axis=0)
    std = np.std(all_preds, axis=0)
    model.eval()
    return mean, std

# -------------------------
# Utility: save plots
# -------------------------
def plot_loss(history, out_path):
    plt.figure(figsize=(6,4))
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_scatter(y_true, y_pred, out_path, title=""):
    plt.figure(figsize=(5,5))
    plt.scatter(y_true[:,0], y_true[:,1], s=6, label="true")
    plt.scatter(y_pred[:,0], y_pred[:,1], s=6, label="pred")
    plt.legend()
    plt.title(title)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_error_hist(errors, out_path):
    plt.figure(figsize=(6,4))
    plt.hist(errors, bins=50)
    plt.xlabel("position error (m)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_model_rmse_bar(results, out_path):
    """
    results: list of dicts with keys 'name' and 'rmse'
    """
    names = [r["name"] for r in results]
    rmses = [r["rmse"] for r in results]

    # sort by RMSE (ascending)
    order = np.argsort(rmses)
    names = [names[i] for i in order]
    rmses = [rmses[i] for i in order]

    plt.figure(figsize=(8,4))
    bars = plt.bar(names, rmses)

    # highlight best model
    bars[0].set_color("tab:green")

    plt.ylabel("RMSE (m)")
    plt.xlabel("Model")
    plt.xticks(rotation=30, ha="right")
    plt.title("Model Performance Comparison (Test Set)")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -------------------------
# Main orchestration
# -------------------------
def main():
    X, y, scaler = load_data()
    print("X shape:", X.shape, "y shape:", y.shape)

    # Train/test split (temporal as before: shuffle=False) but also carve val set for deep models
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    # further split a small val from train for deep models
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.15, shuffle=True, random_state=42)

    print("Train:", len(X_train), "Val:", len(X_val), "Test:", len(X_test))

    results = []

    # -------------------------
    # 1) SKLEARN baselines (fast)
    # -------------------------
    sklearn_models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "RandomForest": RandomForestRegressor(n_estimators=150, n_jobs=-1),
        "SklearnMLP": MLPRegressor(hidden_layer_sizes=(512,256,128), max_iter=400)
    }

    for name, model in sklearn_models.items():
        print(f"\nTraining SKLEARN model: {name}")
        model.fit(X_train_full, y_train_full)
        preds = model.predict(X_test)
        r = {
            "name": name,
            "rmse": rmse(y_test, preds),
            "r2": r2_score(y_test, preds),
            "cos_sim": cos_sim_mean(y_test, preds),
            "model": model,
            "preds": preds
        }
        print(f"{name} RMSE={r['rmse']:.4f}, R2={r['r2']:.4f}")
        results.append(r)
        # save sklearn model
        path = os.path.join(MODEL_DIR, f"{name}.pkl")
        try:
            pickle.dump(model, open(path, "wb"))
        except Exception as e:
            print("Failed to pickle sklearn model:", e)

        # save predictions and plot
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        pd.DataFrame({"x_true": y_test[:,0], "y_true": y_test[:,1],
                      "x_pred": preds[:,0], "y_pred": preds[:,1]}).to_csv(os.path.join(MODEL_DIR, f"{name}_preds_{ts}.csv"), index=False)
        plot_scatter(y_test, preds, os.path.join(PLOTS_DIR, f"{name}_scatter_{ts}.png"), title=name)
        errs = np.linalg.norm(preds - y_test, axis=1)
        plot_error_hist(errs, os.path.join(PLOTS_DIR, f"{name}_errhist_{ts}.png"))

    # -------------------------
    # 2) PyTorch models: MLP, CNN, Hybrid (ConvMLP)
    # -------------------------
    deep_models = {
        "PyMLP": PyMLP,
        "Conv1D": Conv1DNet,
        "ConvMLP": ConvMLPNet
    }
    deep_results = []

    for name, cls in deep_models.items():
        print(f"\nTraining deep model: {name}")
        model = cls(input_dim=X.shape[1], dropout_p=0.3)
        train_ds = LidarDataset(X_train, y_train)
        val_ds = LidarDataset(X_val, y_val)
        model, history = train_torch_model(model, train_ds, val_ds, epochs=70, batch_size=128, lr=1e-4)
        # save loss curve
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_loss(history, os.path.join(PLOTS_DIR, f"{name}_loss_{ts}.png"))
        # test-time predictions (regular deterministic)
        model.eval()
        preds_list = []
        with torch.no_grad():
            loader = DataLoader(LidarDataset(X_test, y_test), batch_size=256)
            for Xb, _ in loader:
                Xb = Xb.to(DEVICE)
                out = model(Xb).cpu().numpy()
                preds_list.append(out)
        preds = np.vstack(preds_list)
        r = {
            "name": name,
            "rmse": rmse(y_test, preds),
            "r2": r2_score(y_test, preds),
            "cos_sim": cos_sim_mean(y_test, preds),
            "model": model,
            "preds": preds,
            "history": history
        }
        print(f"{name} RMSE={r['rmse']:.4f}, R2={r['r2']:.4f}")
        deep_results.append(r)
        results.append(r)

        # Save model weights
        model_path = os.path.join(MODEL_DIR, f"{name}_{ts}.pt")
        torch.save(model.state_dict(), model_path)

        # save predictions and plots
        pd.DataFrame({"x_true": y_test[:,0], "y_true": y_test[:,1],
                      "x_pred": preds[:,0], "y_pred": preds[:,1]}).to_csv(os.path.join(MODEL_DIR, f"{name}_preds_{ts}.csv"), index=False)
        plot_scatter(y_test, preds, os.path.join(PLOTS_DIR, f"{name}_scatter_{ts}.png"), title=name)
        errs = np.linalg.norm(preds - y_test, axis=1)
        plot_error_hist(errs, os.path.join(PLOTS_DIR, f"{name}_errhist_{ts}.png"))

        # compute MC dropout uncertainty (optional small T to check)
        try:
            mean_mc, std_mc = mc_dropout_predict(model, X_test, T=30, batch_size=256)
            # save mean+std to CSV
            df_mc = pd.DataFrame({
                "x_true": y_test[:,0], "y_true": y_test[:,1],
                "x_pred_mean": mean_mc[:,0], "y_pred_mean": mean_mc[:,1],
                "x_std": std_mc[:,0], "y_std": std_mc[:,1],
            })
            df_mc.to_csv(os.path.join(MODEL_DIR, f"{name}_mc_{ts}.csv"), index=False)
            # diagnostic: fraction where true within 2*sigma of mean
            within_2sigma = np.mean(np.linalg.norm(mean_mc - y_test, axis=1) < 2.0 * np.linalg.norm(std_mc, axis=1))
            print(f"MC dropout diagnostic (frac within 2*sigma): {within_2sigma:.3f}")
        except Exception as e:
            print("MC dropout predict failed:", e)

    # -------------------------
    # Choose best by RMSE
    # -------------------------
    best = min(results, key=lambda r: r["rmse"])
    print("\n=== BEST MODEL ===")
    print(best["name"], "RMSE=", best["rmse"])
    bar_path = os.path.join(PLOTS_DIR, f"model_rmse_bar.png")
    plot_model_rmse_bar(results, bar_path)
    print(f"Saved model RMSE bar plot: {bar_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_log = {
        "timestamp": timestamp,
        "models": [
            {
                "name": r["name"],
                "rmse": float(r["rmse"]),
                "r2": float(r["r2"]),
                "cos_sim": float(r["cos_sim"]) if "cos_sim" in r else float(r.get("cos_sim", np.nan))
            } for r in results
        ],
        "best_model": best["name"],
        "best_rmse": float(best["rmse"]),
        "notes": "PyTorch models saved as .pt; sklearn models saved as .pkl"
    }
    with open(os.path.join(MODEL_DIR, f"training_summary_{timestamp}.json"), "w") as fh:
        json.dump(training_log, fh, indent=2)
    print("Saved training summary.")

    # -------------------------
    # Optional: SHAP interpretability for best PyTorch model (if shap installed)
    # -------------------------
    try:
        import shap
        print("Running SHAP (this may be slow) ...")
        # choose small background set
        bg = X_train[np.random.choice(len(X_train), min(200, len(X_train)), replace=False)]
        if isinstance(best["model"], nn.Module):
            # DeepExplainer expects a pytorch model wrapper or use KernelExplainer on predict function
            # We'll use KernelExplainer on a small set to be robust
            def model_predict_np(x_in):
                model = best["model"]
                model.eval()
                with torch.no_grad():
                    xb = torch.tensor(x_in, dtype=torch.float32).to(DEVICE)
                    out = model(xb).cpu().numpy()
                return out
            explainer = shap.KernelExplainer(model_predict_np, bg[:50])
            sample_idx = np.random.choice(len(X_test), min(50, len(X_test)), replace=False)
            shap_values = explainer.shap_values(X_test[sample_idx], nsamples=100)
            # Save one summary plot
            shap.summary_plot(shap_values, X_test[sample_idx], show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, f"shap_summary_{timestamp}.png"))
            plt.close()
            print("Saved SHAP summary.")
        else:
            print("Best model isn't a PyTorch model; SHAP on sklearn models not implemented here.")
    except Exception as e:
        print("SHAP not run (missing or crashed):", e)

    print("Done.")

if __name__ == "__main__":
    main()

