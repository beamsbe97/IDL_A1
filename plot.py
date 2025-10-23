import numpy as np
import matplotlib.pyplot as plt
import json

with open('config.json', 'r') as file:
    config = json.load(file)
plt.figure(figsize=(10,6))
colours = ["blue", "orange", "green"]

ablation = "baseline"
train_losses = np.load(f"training_data/task1_{config["dataset_name"]}_{ablation}_mlp_train_losses.npy")
val_losses = np.load(f"training_data/task1_{config["dataset_name"]}_{ablation}_mlp_val_losses.npy")
plt.plot(train_losses, label=f"Train (Baseline)", color="blue")
plt.plot(val_losses, label=f"Val (Baseline)", color="blue",linestyle="--")

ablation = "shallow"
train_losses = np.load(f"training_data/task1_{config["dataset_name"]}_{ablation}_mlp_train_losses.npy")
val_losses = np.load(f"training_data/task1_{config["dataset_name"]}_{ablation}_mlp_val_losses.npy")
plt.plot(train_losses, label=f"Train (Shallow)", color="orange")
plt.plot(val_losses, label=f"Val (Shallow)", color="orange",linestyle="--")

ablation = "deep"
train_losses = np.load(f"training_data/task1_{config["dataset_name"]}_{ablation}_mlp_train_losses.npy")
val_losses = np.load(f"training_data/task1_{config["dataset_name"]}_{ablation}_mlp_val_losses.npy")
plt.plot(train_losses, label=f"Train (Deep)", color="green")
plt.plot(val_losses, label=f"Val (Deep)", color="green",linestyle="--")


plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MLP Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig(f"mlp_NetworkSize.png", dpi=300)
plt.close()
