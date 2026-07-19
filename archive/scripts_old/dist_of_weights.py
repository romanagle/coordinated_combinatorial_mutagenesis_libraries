import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

reg_values = ["22", "24", "42", "45", "82", "84", "210", "410", "810", "1012", "1214"]

for v in reg_values:
    plt.figure(figsize=(6,4))
    weights = np.load(f'/home/nagle/final_version/output/reg_param_ism_graphs/pairwise_weights/{v}/weights_epoch_3.npy')
    values = np.abs(weights.flatten())
    sns.kdeplot(values, bw_adjust=0.7, fill=True)
    plt.savefig(f'/home/nagle/final_version/output/reg_param_ism_graphs/pairwise_weights/distofweights/{v}distofweights.png')

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for e in range(0,15,5):
    weights = np.load(f'/home/nagle/final_version/output/reg_param_ism_graphs/pairwise_weights/1214/weights_epoch_{e}.npy')
    values = weights.flatten()
    sns.kdeplot(
        np.abs(values),
        bw_adjust=0.7,
        fill=True,
        alpha=0.4,
        ax=axes[0],
        label=f"epoch {e}")
    sns.kdeplot(
        values,
        bw_adjust=0.7,
        fill=True,
        alpha=0.4,
        ax=axes[1],
        label=f"epoch {e}")

# Left panel
axes[0].set_title("Absolute weights")
axes[0].set_xlabel("Abs(weight)")
axes[0].set_xlim(0, 0.5)

# Right panel
axes[1].set_title("Weights")
axes[1].set_xlabel("Weight value")
axes[1].set_xlim(-0.5, 0.5)

# Shared y-label
fig.supylabel("Density")

# Legends
axes[0].legend(title="Epoch")
axes[1].legend(title="Epoch")

plt.tight_layout()
plt.savefig("/home/nagle/final_version/output/reg_param_ism_graphs/pairwise_weights/distofweights/1214_abs_vs_signed_panels.png")
plt.close()



for v in reg_values:
    plt.figure(figsize=(6,4))
    weights = np.load(f'/home/nagle/final_version/output/reg_param_ism_graphs/additive_weights/{v}/weights_epoch_3.npy')
    values = np.abs(weights.flatten())
    sns.kdeplot(values, bw_adjust=0.7, fill=True)
    plt.savefig(f'/home/nagle/final_version/output/reg_param_ism_graphs/additive_weights/distofweights/{v}distofweights.png')

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for e in range(0,15,5):
    weights = np.load(f'/home/nagle/final_version/output/reg_param_ism_graphs/additive_weights/1214/weights_epoch_{e}.npy')
    values = weights.flatten()
    sns.kdeplot(
        np.abs(values),
        bw_adjust=0.7,
        fill=True,
        alpha=0.4,
        ax=axes[0],
        label=f"epoch {e}")
    sns.kdeplot(
        values,
        bw_adjust=0.7,
        fill=True,
        alpha=0.4,
        ax=axes[1],
        label=f"epoch {e}")

# Left panel
axes[0].set_title("Absolute weights")
axes[0].set_xlabel("Abs(weight)")

# Right panel
axes[1].set_title("Weights")
axes[1].set_xlabel("Weight value")

# Shared y-label
fig.supylabel("Density")

# Legends
axes[0].legend(title="Epoch")
axes[1].legend(title="Epoch")

plt.tight_layout()
plt.savefig("/home/nagle/final_version/output/reg_param_ism_graphs/additive_weights/distofweights/1214_abs_vs_signed_panels.png")
plt.close()


