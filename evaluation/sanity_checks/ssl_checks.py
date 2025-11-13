import tensorflow as tf
import numpy as np
from sklearn.manifold import TSNE
from itertools import combinations
from evaluation.sanity_checks.visualization_utils import plot_tsne
from evaluation.sanity_checks.logger import DiagnosticsLogger

def embedding_diagnostics(model, dataset, log_dir=None, num_batches=2, step=0):
    """Compute intra/inter distances and log plots."""
    embeddings, labels = [], []
    for i, (x, y) in enumerate(dataset.take(num_batches)):
        outputs = model.forward(x, training=False)
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]
        embeddings.append(outputs.numpy())
        labels.append(y.numpy())
    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)

    class_centroids = {c: embeddings[labels == c].mean(0) for c in np.unique(labels)}
    intra = np.mean([
        np.linalg.norm(embeddings[labels == c] - class_centroids[c], axis=1).mean()
        for c in class_centroids
    ])
    inter = np.mean([
        np.linalg.norm(class_centroids[a] - class_centroids[b])
        for a, b in combinations(class_centroids.keys(), 2)
    ])
    ratio = inter / intra
    print(f"[SupConCheck] Intra={intra:.4f}, Inter={inter:.4f}, Ratio={ratio:.2f}")

    # Log metrics and TSNE visualization
    if log_dir:
        logger = DiagnosticsLogger(log_dir)
        logger.log_scalar("ssl/intra_class_distance", intra, step)
        logger.log_scalar("ssl/inter_class_distance", inter, step)
        logger.log_scalar("ssl/inter_intra_ratio", ratio, step)

        tsne = TSNE(n_components=2, perplexity=30)
        reduced = tsne.fit_transform(embeddings)
        fig = plot_tsne(reduced, labels, title="SupCon Embeddings")
        logger.log_figure("ssl/embedding_tsne", fig, step)
        logger.close()
