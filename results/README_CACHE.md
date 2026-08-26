# results/ — CACHE MANIFEST

## ⚠️ CRITICAL — DO NOT DELETE, DO NOT `git clean -fd`

The `.npz` files in this directory are **cached Chronos-2 GPU predictions**.
They are expensive to regenerate (full model inference over all val/test
windows on 4 datasets) and are what make the four ETT baselines reproducible
without re-running the backbone.

`results/` is in `.gitignore`, so these files are **NOT tracked by git** and a
`git clean` or fresh clone will lose them. Preserve them on disk.

## Critical caches (Chronos-2 predictions)

| File | Content |
|---|---|
| `ETTh1_chronos.npz` / `ETTh2_chronos.npz` / `ETTm1_chronos.npz` / `ETTm2_chronos.npz` | Chronos-2 point forecasts (val + test) per dataset |
| `ETTh1_quantiles.npz` / `ETTh2_quantiles.npz` / `ETTm1_quantiles.npz` / `ETTm2_quantiles.npz` | Chronos-2 native quantiles (0.01–0.99), per dataset |
| `forecasts.npz` / `val_forecasts.npz` / `future_stats.npz` | intermediate forecast / val / future-stat caches |

## Reference baselines these caches reproduce

| Dataset | Chronos-2 test MSE | Chronos-2 test MAE |
|---|---|---|
| ETTh1 | 6.1968 | 1.8649 |
| ETTh2 | 26.9971 | 3.8534 |
| ETTm1 | 3.0576 | 1.2440 |
| ETTm2 | 16.2771 | 2.7447 |

## How to regenerate (if a cache is lost)

```bash
python run_temporal_ablation_all_ett.py --force   # regenerates *_chronos.npz
# or, for quantiles: rerun the uncertainty diagnostic's caching step
```

## Everything else in results/

`.csv` results, `.png` plots, and `run_*.log` logs are all regenerable from the
scripts and are lower priority — but still worth keeping for paper-writing
(including negative results).
