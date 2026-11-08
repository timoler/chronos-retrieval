"""Train on aligned softmax forecasts and compute both hard Top-5 forecasts."""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch

from .data import activations, slot
from .model import ConditionalExpertMLPRetriever


@dataclass(frozen=True)
class Training:
    epochs: int = 30
    batch_size: int = 64
    candidates: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 0.0001


class TextVectors:
    """Read the original per-text .npy cache, or encode text using a local model."""
    def __init__(self, cache=None, encoder=None):
        self.cache = Path(cache) if cache is not None else None
        self.encoder_path = Path(encoder) if encoder is not None else None
        self.encoder = None
        self.values = {}

    def __call__(self, text):
        if text in self.values:
            return self.values[text]
        key = hashlib.sha256(json.dumps(text, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        path = self.cache / f"{key}.npy" if self.cache is not None else None
        if path is not None and path.is_file():
            value = np.load(path, allow_pickle=False)
        elif self.encoder_path is not None:
            if self.encoder is None:
                from sentence_transformers import SentenceTransformer
                self.encoder = SentenceTransformer(str(self.encoder_path.resolve()), device="cpu",
                                                  local_files_only=True)
            value = self.encoder.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]
        else:
            raise FileNotFoundError(f"Missing embedding {key}.npy; supply the cache or a local encoder.")
        value = np.asarray(value, dtype=np.float32)
        if value.shape != (384,) or not np.isfinite(value).all() or not np.isclose(np.linalg.norm(value), 1, atol=1e-4):
            raise ValueError("Expected a finite unit-length 384-dimensional text vector")
        self.values[text] = value
        return value


def init_model(pool, vector, context, seed, device, drop=None):
    matrix = np.zeros((8, 384), dtype=np.float32)
    for e in pool:
        if e["id"] != drop:
            matrix[slot(e)] = vector(e["mechanism"])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    model = ConditionalExpertMLPRetriever(context, 384, 8).to(device)
    model.set_expert_embeddings(matrix)
    return model


def fit(stage, pool, vectors, seed, device, training=Training(), drop=None):
    length = stage["X"]["bank"].shape[1]
    act = activations(stage, pool, drop)
    model = init_model(pool, vectors, length, seed, device, drop)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay)
    rng = np.random.default_rng(seed)
    for _ in range(training.epochs):
        model.train()
        order = rng.permutation(len(stage["fit"]))
        for offset in range(0, len(order), training.batch_size):
            rows = order[offset:offset + training.batch_size]
            n = len(stage["bank"])
            cand = rng.choice(n, min(training.candidates, n), replace=False)
            xq = torch.as_tensor(stage["X"]["fit"][rows], device=device)
            xc = torch.as_tensor(stage["X"]["bank"][cand], device=device)
            aq = torch.as_tensor(act["fit"][rows], device=device)
            ac = torch.as_tensor(act["bank"][cand], device=device)
            shape = torch.as_tensor(stage["R"]["fit"][rows] @ stage["R"]["bank"][cand].T / length, device=device)
            score = model.combine(model.delta(model.encode(xq, aq), model.encode(xc, ac)), shape, "full")
            weights = torch.softmax(score, dim=1)
            future = torch.as_tensor(stage["Y"]["bank"][cand], device=device)
            aligned = future[None, :, :] + xq.mean(1)[:, None, None] - xc.mean(1)[None, :, None]
            pred = (weights[:, :, None] * aligned).sum(1)
            target = torch.as_tensor(stage["Y"]["fit"][rows], device=device)
            loss = torch.nn.functional.mse_loss(pred, target)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite fit MSE")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return model


@torch.no_grad()
def predict(model, stage, pool, drop=None):
    model.eval()
    device = next(model.parameters()).device
    act = activations(stage, pool, drop)
    bank = stage["X"]["bank"]
    query = stage["X"]["query"]
    bank_res = torch.as_tensor(stage["R"]["bank"], device=device)
    query_res = torch.as_tensor(stage["R"]["query"], device=device)
    def encode(rows, states):
        return model.encode(torch.as_tensor(rows, device=device), torch.as_tensor(states, device=device))
    zb = torch.cat([encode(bank[i:i+512], act["bank"][i:i+512]) for i in range(0, len(bank), 512)])
    neighbors = []
    for start in range(0, len(query), 32):
        zq = encode(query[start:start+32], act["query"][start:start+32])
        scores = torch.full((len(zq), 5), -torch.inf, device=device)
        ids = torch.zeros_like(scores, dtype=torch.long)
        for b in range(0, len(bank), 1024):
            block = zb[b:b+1024]
            shape = query_res[start:start+32] @ bank_res[b:b+1024].T / bank.shape[1]
            score = model.combine(model.delta(zq, block), shape, "full")
            ix = torch.arange(b, b + len(block), device=device).expand(len(zq), -1)
            joint = torch.cat((scores, score), 1)
            joint_ids = torch.cat((ids, ix), 1)
            scores, pos = torch.topk(joint, 5, dim=1)
            ids = torch.gather(joint_ids, 1, pos)
        neighbors.append(ids.cpu().numpy())
    idx = np.concatenate(neighbors)
    future = stage["Y"]["bank"].astype(np.float64)
    if not np.all(stage["bank"][idx] + future.shape[1] <= stage["query"][:, None]):
        raise ValueError("Selected a candidate whose future is unavailable at query time")
    q_mean = query.astype(np.float64).mean(1)
    b_mean = bank.astype(np.float64).mean(1)
    return dict(indices=idx, starts=stage["query"], raw=future[idx].mean(1),
                aligned=(future[idx] + q_mean[:, None, None] - b_mean[idx][:, :, None]).mean(1),
                coverage={e["id"]: float((act["query"][:,slot(e)] >= 0.5).mean()) for e in pool})


def mse(target, pred):
    return float(np.mean((np.asarray(target, np.float64) - np.asarray(pred, np.float64)) ** 2))


def select_weight(val_y, val_base, val_mem):
    arrays = [np.asarray(a, dtype=np.float64) for a in (val_y, val_base, val_mem)]
    if any(a.ndim != 2 or a.shape != arrays[0].shape or not np.isfinite(a).all() for a in arrays):
        raise ValueError("Expected three finite VAL arrays with identical [windows, horizon] shape.")
    target, base, mem = arrays
    grid = (0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1)
    return min(grid, key=lambda w: mse(target, (1-w)*base + w*mem))
