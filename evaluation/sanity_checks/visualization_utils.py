import matplotlib.pyplot as plt
import numpy as np

def plot_tsne(embeddings, labels, title="Embedding t-SNE"):
    fig, ax = plt.subplots(figsize=(6,5))
    classes = np.unique(labels)
    for c in classes:
        idx = labels == c
        ax.scatter(embeddings[idx,0], embeddings[idx,1], label=f"Class {c}", s=15)
    ax.legend()
    ax.set_title(title)
    return fig

def plot_reconstructions(x_batch, x_hat, num_samples=5, title="Reconstructions"):
    n = min(num_samples, x_batch.shape[0])
    fig, axes = plt.subplots(2, n, figsize=(n*2, 4))
    for i in range(n):
        axes[0, i].imshow(np.clip(x_batch[i], 0, 1))
        axes[0, i].axis("off")
        axes[1, i].imshow(np.clip(x_hat[i], 0, 1))
        axes[1, i].axis("off")
    fig.suptitle(title)
    return fig
