import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

def plot_prototypes(z_support, y_support, z_query, y_query, prototypes):
    """Visualize embedding space with prototypes (for ProtoNet)."""
    tsne = TSNE(n_components=2, perplexity=20)
    combined = np.concatenate([z_support, z_query, prototypes], axis=0)
    reduced = tsne.fit_transform(combined)
    n_s, n_q, n_p = len(z_support), len(z_query), len(prototypes)

    fig, ax = plt.subplots(figsize=(6,6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(np.unique(y_support))))
    for i, c in enumerate(np.unique(y_support)):
        idx_s = np.where(y_support == c)[0]
        idx_q = np.where(y_query == c)[0]
        ax.scatter(reduced[idx_s,0], reduced[idx_s,1], color=colors[i], label=f"S-{c}", s=25, marker='o')
        ax.scatter(reduced[n_s+idx_q,0], reduced[n_s+idx_q,1], color=colors[i], label=f"Q-{c}", s=25, marker='x')
    proto_coords = reduced[n_s+n_q:]
    ax.scatter(proto_coords[:,0], proto_coords[:,1], color='black', s=100, marker='*', label='Prototypes')
    ax.legend()
    ax.set_title("Prototypical Network Embeddings")
    return fig

def plot_meta_adaptation(losses):
    """Simple line plot of adaptation loss across episodes."""
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(losses, '-o')
    ax.set_title("Meta Adaptation Loss Across Episodes")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss")
    return fig
