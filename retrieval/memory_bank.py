"""Historical memory bank: sliding-window (context, future) pairs.

Built only from the training series so no future information from
validation/test leaks into the memory.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class MemoryBank:
    contexts: np.ndarray  # (n_memories, context_length)
    futures: np.ndarray   # (n_memories, prediction_length)

    @property
    def n_memories(self) -> int:
        return self.contexts.shape[0]

    @classmethod
    def from_series(cls, series, context_length, prediction_length):
        series = np.asarray(series, dtype=np.float32)
        window = context_length + prediction_length
        if len(series) < window:
            raise ValueError(f"series too short: {len(series)} < {window}")

        n = len(series) - window + 1
        contexts = np.stack([series[i : i + context_length] for i in range(n)])
        futures = np.stack([series[i + context_length : i + window] for i in range(n)])
        return cls(contexts=contexts, futures=futures)
