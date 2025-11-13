import tensorflow as tf
import numpy as np
import random

class EpisodicLoader:
    """Generates N-way-K-shot episodes for meta-learning."""

    def __init__(self, dataset, n_way=5, k_shot=1, q_query=15):
        self.dataset = list(dataset.as_numpy_iterator())
        self.n_way, self.k_shot, self.q_query = n_way, k_shot, q_query

    def sample_episode(self):
        classes = random.sample(list({y for _, y in self.dataset}), self.n_way)
        support, query = [], []
        for c in classes:
            samples = [(x, y) for x, y in self.dataset if y == c]
            chosen = random.sample(samples, self.k_shot + self.q_query)
            support += chosen[:self.k_shot]
            query   += chosen[self.k_shot:]
        s_x, s_y = zip(*support)
        q_x, q_y = zip(*query)
        return (np.stack(s_x), np.array(s_y), np.stack(q_x), np.array(q_y))
