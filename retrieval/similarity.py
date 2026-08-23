"""Similarity measures between a query and memory contexts."""

import numpy as np


def _standardize(x):
    centered = x - x.mean(axis=1, keepdims=True)
    std = centered.std(axis=1, keepdims=True)
    # zero-variance rows become all-zero, yielding correlation 0 with everything
    return centered / np.where(std == 0, 1.0, std)


def pearson_similarity_matrix(queries, contexts):
    queries = np.asarray(queries, dtype=np.float64)
    contexts = np.asarray(contexts, dtype=np.float64)
    q = _standardize(queries)
    c = _standardize(contexts)
    return (c @ q.T / queries.shape[1]).T  # (n_queries, n_contexts)


def pearson_similarity(query, contexts):
    return pearson_similarity_matrix(query[None, :], contexts)[0]


def cosine_similarity_matrix(queries, contexts):
    queries = np.asarray(queries, dtype=np.float64)
    contexts = np.asarray(contexts, dtype=np.float64)
    denom = np.outer(np.linalg.norm(queries, axis=1), np.linalg.norm(contexts, axis=1))
    return np.where(denom == 0, 0.0, (queries @ contexts.T) / denom)  # (n_queries, n_contexts)


def cosine_similarity(query, contexts):
    return cosine_similarity_matrix(query[None, :], contexts)[0]
