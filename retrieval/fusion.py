"""Forecast fusion: softmax weighting over retrieved futures and lambda blending."""

import numpy as np


def softmax(x, temperature=1.0):
    x = np.asarray(x, dtype=np.float64) / temperature
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def memory_forecast(similarity_scores, futures, temperature=1.0):
    weights = softmax(similarity_scores, temperature)
    return weights @ futures


def fuse(chronos_forecast, memory_fcst, lambda_):
    return (1.0 - lambda_) * np.asarray(chronos_forecast) + lambda_ * np.asarray(memory_fcst)
