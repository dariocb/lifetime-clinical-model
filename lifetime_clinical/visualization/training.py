from __future__ import annotations
import matplotlib.pyplot as plt


def plot_fit_history(model, ax=None, title=None):
    history = getattr(model, "fit_history", [])
    if not history:
        raise ValueError("Model has no fit_history; call fit first")
    ax = ax or plt.subplots(figsize=(6.4, 3.2))[1]
    epochs = [row["epoch"] for row in history]
    train_key = "train_nll" if "train_nll" in history[0] else "train_objective"
    monitor_key = "monitor_nll" if "monitor_nll" in history[0] else "monitor_objective"
    ax.plot(epochs, [row[train_key] for row in history], label="train")
    ax.plot(epochs, [row[monitor_key] for row in history], label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("objective (lower is better)")
    ax.set_title(title or getattr(model, "model_name", "fit"))
    ax.legend(frameon=False)
    ax.grid(alpha=.2)
    return ax
