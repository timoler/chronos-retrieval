"""Top-k retrieval over a memory bank."""

import numpy as np

from .similarity import pearson_similarity, pearson_similarity_matrix


def _top_k_indices(scores, top_k):
    top_k = min(top_k, scores.shape[-1])
    idx = np.argpartition(scores, -top_k, axis=-1)[..., -top_k:]
    order = np.argsort(-np.take_along_axis(scores, idx, axis=-1), axis=-1)
    return np.take_along_axis(idx, order, axis=-1)


def retrieve(query, memory_bank, top_k, similarity_fn=pearson_similarity):
    scores = similarity_fn(query, memory_bank.contexts)
    idx = _top_k_indices(scores, top_k)
    return idx, scores[idx], memory_bank.futures[idx]


def retrieve_batch(queries, memory_bank, top_k, similarity_fn=pearson_similarity_matrix):
    scores = similarity_fn(queries, memory_bank.contexts)  # (n_queries, n_memories)
    idx = _top_k_indices(scores, top_k)
    return idx, np.take_along_axis(scores, idx, axis=-1), memory_bank.futures[idx]
