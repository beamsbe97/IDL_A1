import numpy as np
import matplotlib.pyplot as plt

# Load the losses
train_losses = np.load("task1_mlp_train_losses.npy")
val_losses = np.load("task1_mlp_val_losses.npy")

# Plot learning curves
plt.figure(figsize=(8,5))
plt.plot(train_losses, label="Train Loss", color="blue")
plt.plot(val_losses, label="Validation Loss", color="orange")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MLP Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig("mlp_learning_curve.png", dpi=300)
plt.close()
