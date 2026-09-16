#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Learned Critic v1 — LODO validation on the combined 94-sample knowledge-effect dataset.

Reads results/learned_critic_dataset_v2/knowledge_effect_samples_combined.parquet
(48 v1 samples + 46 v2 samples, 10 datasets) and trains a small Learned Critic with
Leave-One-Dataset-Out evaluation — identical method to v0, now with 10 folds.

Constraints (unchanged):
- No forecasting rerun; Critic v6.1 frozen; TEST never read.
- Regression target = mean_relative_gain; auxiliary label = contribution_label.
- Features: low-dim generic + activation + critic-audit only (no dataset_id/text-embedding).
- Per-fold imputer+scaler fit on TRAIN only.
- Models: Ridge / RandomForest / small MLP (3 seeds). Baselines: always_positive / v6.1.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn

IN_PARQUET = Path("results/learned_critic_dataset_v2/knowledge_effect_samples_combined.parquet")
OUT_ROOT = Path("results/learned_critic_v1")

MLP_SEEDS = [42, 123, 2026]
RF_SEED = 42
MLP_EPOCHS = 3000
MLP_LR = 1e-3
MLP_WD = 1e-4

GENERIC_FEATURES = [
    "target_mean", "target_std", "normalized_mean_abs_diff", "lag1_autocorrelation",
    "trend_strength", "nonstationarity_score", "dominant_period", "dominant_period_strength",
    "spectral_entropy", "context_length", "forecast_horizon",
]
ACTIVATION_FEATURES = [
    "activation_mean", "activation_std", "activation_zero_ratio",
    "activation_nonzero_ratio", "activation_unique_ratio",
    "activation_block_mean_std", "activation_temporal_stability",
]
COVERAGE_MAP = {"DEAD": 0.0, "CONSTANT_LOW_INFORMATION": 1.0, "ACCEPTED_LOCAL_SCOPE": 2.0, "OK": 3.0}
AUDIT_FEATURES = [
    "v6_1_final_decision", "observability_decision", "evidence_decision",
    "fidelity_decision", "generalization_decision", "redundancy_decision",
    "activation_coverage_decision", "compiler_decision", "repair_happened",
]


def encode_audit(df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    out["v6_1_final_decision"] = (df["v6_1_final_decision"] == "ACCEPT").astype(float)
    out["observability_decision"] = (df["observability_decision"] == "PASS").astype(float)
    out["evidence_decision"] = (df["evidence_decision"] == "PASS").astype(float)
    out["fidelity_decision"] = (df["fidelity_decision"] == "PASS").astype(float)
    out["generalization_decision"] = (df["generalization_decision"] == "PASS").astype(float)
    out["redundancy_decision"] = (df["redundancy_decision"] == "DUPLICATE_ACTIVATION").astype(float)
    out["activation_coverage_decision"] = df["activation_coverage_decision"].map(COVERAGE_MAP).astype(float)
    out["compiler_decision"] = (df["compiler_decision"] == "COMPILED").astype(float)
    out["repair_happened"] = df["repair_happened"].astype(float)
    return pd.DataFrame(out)


def reg_metrics(y, p):
    return {"mae": float(mean_absolute_error(y, p)),
            "rmse": float(np.sqrt(mean_squared_error(y, p))),
            "spearman": float(spearmanr(y, p)[0]) if len(np.unique(p)) > 1 else float("nan")}


def clf_metrics(y_true_bin, y_pred_bin):
    acc = float(accuracy_score(y_true_bin, y_pred_bin))
    p, r, f, _ = precision_recall_fscore_support(y_true_bin, y_pred_bin, pos_label=1, zero_division=0, average="binary")
    return {"accuracy": acc, "precision": float(p), "recall": float(r), "f1": float(f)}


def to_binary_pred(pred_cont):
    return (pred_cont > 0).astype(int)


class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(Xtr, ytr, Xte, seed, epochs=MLP_EPOCHS, lr=MLP_LR, wd=MLP_WD):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)
    model = MLP(Xtr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    lossfn = nn.MSELoss()
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = lossfn(model(Xtr_t), ytr_t)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        return model(Xtr_t).numpy(), model(Xte_t).numpy()


def load_data():
    df = pd.read_parquet(IN_PARQUET)
    audit = encode_audit(df)
    feature_cols = GENERIC_FEATURES + ACTIVATION_FEATURES + AUDIT_FEATURES
    X = pd.DataFrame(index=df.index)
    for c in GENERIC_FEATURES + ACTIVATION_FEATURES:
        X[c] = df[c].astype(float)
    for c in AUDIT_FEATURES:
        X[c] = audit[c].astype(float)
    X = X[feature_cols].to_numpy(dtype=float)
    datasets = sorted(df["dataset_name"].unique().tolist())
    return {"X": X, "y": df["mean_relative_gain"].to_numpy(float),
            "label": df["contribution_label"].to_numpy(),
            "dataset": df["dataset_name"].to_numpy(),
            "expert_id": df["expert_id"].to_numpy(),
            "v6_1_decision": df["v6_1_final_decision"].to_numpy(),
            "feature_cols": feature_cols, "datasets": datasets}


def build_folds(data):
    folds = []
    for held in data["datasets"]:
        te_idx = np.where(data["dataset"] == held)[0]
        tr_idx = np.where(data["dataset"] != held)[0]
        folds.append((held, tr_idx, te_idx))
    return folds


def preprocess_fold(Xtr, Xte):
    tr_std = Xtr.std(axis=0)
    keep = tr_std > 1e-12
    Xtr_k, Xte_k = Xtr[:, keep], Xte[:, keep]
    imp = SimpleImputer(strategy="median")
    Xtr_i = imp.fit_transform(Xtr_k)
    Xte_i = imp.transform(Xte_k)
    scaler = StandardScaler()
    return scaler.fit_transform(Xtr_i), scaler.transform(Xte_i), int(keep.sum())


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    data = load_data()
    folds = build_folds(data)
    print(f"training samples={len(data['y'])} features={data['X'].shape[1]} folds={len(folds)}", flush=True)

    rows, fold_metrics, fold_mlp_seed_stats = [], [], []
    for held, tr_idx, te_idx in folds:
        Xtr_raw, Xte_raw = data["X"][tr_idx], data["X"][te_idx]
        ytr, yte = data["y"][tr_idx], data["y"][te_idx]
        Xtr, Xte, n_feat = preprocess_fold(Xtr_raw, Xte_raw)

        ridge = Ridge(alpha=1.0).fit(Xtr, ytr)
        ridge_te = ridge.predict(Xte)
        rf = RandomForestRegressor(n_estimators=500, random_state=RF_SEED).fit(Xtr, ytr)
        rf_te = rf.predict(Xte)

        mlp_te_seeds = []
        for s in MLP_SEEDS:
            _, te_pred = train_mlp(Xtr, ytr, Xte, seed=s)
            mlp_te_seeds.append(te_pred)
        mlp_te = np.mean(mlp_te_seeds, axis=0)

        seed_regs = [reg_metrics(yte, p) for p in mlp_te_seeds]
        fold_mlp_seed_stats.append({
            "dataset": held,
            "mae_mean": float(np.mean([r["mae"] for r in seed_regs])),
            "mae_std": float(np.std([r["mae"] for r in seed_regs])),
            "rmse_mean": float(np.mean([r["rmse"] for r in seed_regs])),
            "rmse_std": float(np.std([r["rmse"] for r in seed_regs])),
            "spearman_mean": float(np.nanmean([r["spearman"] for r in seed_regs])),
            "spearman_std": float(np.nanstd([r["spearman"] for r in seed_regs])),
        })

        for k, idx in enumerate(te_idx):
            rows.append({"fold": held, "dataset_name": data["dataset"][idx],
                         "expert_id": data["expert_id"][idx], "true_label": data["label"][idx],
                         "true_mean_relative_gain": float(yte[k]),
                         "ridge_pred": float(ridge_te[k]), "rf_pred": float(rf_te[k]), "mlp_pred": float(mlp_te[k]),
                         "v6_1_decision": data["v6_1_decision"][idx]})

        regs = {"ridge": reg_metrics(yte, ridge_te), "rf": reg_metrics(yte, rf_te), "mlp": reg_metrics(yte, mlp_te)}
        mask = (data["label"][te_idx] != "MIXED")
        y_true_bin = (data["label"][te_idx][mask] == "POSITIVE").astype(int)
        clfs = {}
        for mname, pte in [("ridge", ridge_te), ("rf", rf_te), ("mlp", mlp_te)]:
            clfs[mname] = clf_metrics(y_true_bin, to_binary_pred(pte[mask]))
        clfs["always_positive"] = clf_metrics(y_true_bin, np.ones_like(y_true_bin))
        v6_bin = (data["v6_1_decision"][te_idx][mask] == "ACCEPT").astype(int)
        clfs["v6_1"] = clf_metrics(y_true_bin, v6_bin)

        for mname in ["ridge", "rf", "mlp"]:
            fold_metrics.append({"fold": held, "model": mname, "n_test": int(len(yte)), "n_class": int(mask.sum()),
                                 **regs[mname], **clfs[mname]})
        for bname in ["always_positive", "v6_1"]:
            fold_metrics.append({"fold": held, "model": bname, "n_test": int(len(yte)), "n_class": int(mask.sum()),
                                 "mae": float("nan"), "rmse": float("nan"), "spearman": float("nan"), **clfs[bname]})
        print(f"  [fold {held}] n={len(yte)} | rf MAE={regs['rf']['mae']:.4f} acc={clfs['rf']['accuracy']:.3f} "
              f"v6.1 acc={clfs['v6_1']['accuracy']:.3f}", flush=True)

    pred_df = pd.DataFrame(rows)
    fm_df = pd.DataFrame(fold_metrics)

    pooled = {}
    sub = pred_df[pred_df["true_label"] != "MIXED"]
    y_true_bin = (sub["true_label"] == "POSITIVE").astype(int).to_numpy()
    for mname in ["ridge", "rf", "mlp"]:
        pooled[mname] = clf_metrics(y_true_bin, to_binary_pred(sub[f"{mname}_pred"].to_numpy()))
    pooled["always_positive"] = clf_metrics(y_true_bin, np.ones_like(y_true_bin))
    pooled["v6_1"] = clf_metrics(y_true_bin, (sub["v6_1_decision"] == "ACCEPT").astype(int).to_numpy())

    pooled_reg = {m: reg_metrics(pred_df["true_mean_relative_gain"].to_numpy(), pred_df[f"{m}_pred"].to_numpy())
                  for m in ["ridge", "rf", "mlp"]}

    agg = {}
    for mname in ["ridge", "rf", "mlp"]:
        m = fm_df[fm_df["model"] == mname]
        agg[mname] = {"mae_mean": float(m["mae"].mean()), "mae_std": float(m["mae"].std()),
                      "rmse_mean": float(m["rmse"].mean()), "rmse_std": float(m["rmse"].std()),
                      "spearman_mean": float(m["spearman"].mean()), "spearman_std": float(m["spearman"].std())}

    # train MAE (overfitting check)
    train_mae = {m: [] for m in ["ridge", "rf", "mlp"]}
    for held, tr_idx, te_idx in folds:
        Xtr_raw, Xte_raw = data["X"][tr_idx], data["X"][te_idx]
        ytr, yte = data["y"][tr_idx], data["y"][te_idx]
        Xtr, Xte, _ = preprocess_fold(Xtr_raw, Xte_raw)
        ridge = Ridge(alpha=1.0).fit(Xtr, ytr)
        rf = RandomForestRegressor(n_estimators=500, random_state=RF_SEED).fit(Xtr, ytr)
        mlp_tr = np.mean([train_mlp(Xtr, ytr, Xtr, seed=s)[0] for s in MLP_SEEDS], axis=0)
        for mname, trp in [("ridge", ridge.predict(Xtr)), ("rf", rf.predict(Xtr)), ("mlp", mlp_tr)]:
            train_mae[mname].append(mean_absolute_error(ytr, trp))
    train_mae_agg = {m: float(np.mean(v)) for m, v in train_mae.items()}

    # ---- write outputs ----
    pred_df.to_csv(OUT_ROOT / "lodo_predictions.csv", index=False, encoding="utf-8")
    fm_df.to_csv(OUT_ROOT / "lodo_metrics.csv", index=False, encoding="utf-8")

    comp_rows = []
    for mname in ["ridge", "rf", "mlp"]:
        comp_rows.append({"model": mname, **agg[mname],
                          "pooled_mae": pooled_reg[mname]["mae"], "pooled_rmse": pooled_reg[mname]["rmse"],
                          "pooled_spearman": pooled_reg[mname]["spearman"],
                          "pooled_accuracy": pooled[mname]["accuracy"], "pooled_f1": pooled[mname]["f1"],
                          "pooled_precision": pooled[mname]["precision"], "pooled_recall": pooled[mname]["recall"],
                          "train_mae": train_mae_agg[mname]})
    for bname in ["always_positive", "v6_1"]:
        comp_rows.append({"model": bname, "pooled_accuracy": pooled[bname]["accuracy"],
                          "pooled_f1": pooled[bname]["f1"], "pooled_precision": pooled[bname]["precision"],
                          "pooled_recall": pooled[bname]["recall"]})
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(OUT_ROOT / "model_comparison.csv", index=False, encoding="utf-8")

    feature_list = {"target": "mean_relative_gain", "label": "contribution_label",
                    "n_training_samples": int(len(data["y"])), "evaluation": "leave_one_dataset_out",
                    "n_datasets": len(folds), "datasets": data["datasets"],
                    "forbidden_features": ["dataset_name", "expert_id", "contribution_label", "mean_delta",
                                           "full_raw_mse_*", "without_i_mse_*", "delta_seed*", "relative_gain_seed*",
                                           "text embedding", "TEST set"],
                    "features": {"A_dataset_generic": GENERIC_FEATURES,
                                 "B_expert_activation": ACTIVATION_FEATURES,
                                 "C_critic_audit": AUDIT_FEATURES},
                    "n_raw_features": int(data["X"].shape[1])}
    (OUT_ROOT / "feature_list.json").write_text(json.dumps(feature_list, ensure_ascii=False, indent=2), encoding="utf-8")

    write_report(agg, pooled, pooled_reg, train_mae_agg, fm_df, fold_mlp_seed_stats, data)


def write_report(agg, pooled, pooled_reg, train_mae_agg, fm_df, seed_stats, data):
    L = ["# Learned Critic v1 — LODO 验证报告（合并 94 样本）", "", "## 结果", ""]
    L.append("### 1. 每个模型 10-fold LODO 回归指标 (MAE/RMSE/Spearman, mean±std)")
    L.append("| model | MAE (mean±std) | RMSE (mean±std) | Spearman (mean±std) |")
    L.append("|---|---|---|---|")
    for m in ["ridge", "rf", "mlp"]:
        a = agg[m]
        L.append(f"| {m} | {a['mae_mean']:.4f} ± {a['mae_std']:.4f} | {a['rmse_mean']:.4f} ± {a['rmse_std']:.4f} | "
                 f"{a['spearman_mean']:.3f} ± {a['spearman_std']:.3f} |")
    L.append("")
    L.append("### 2. Positive/Negative 二分类 (pooled over folds, MIXED 不参与)")
    L.append("| model | accuracy | precision | recall | F1 |")
    L.append("|---|---|---|---|---|")
    for m in ["ridge", "rf", "mlp", "always_positive", "v6_1"]:
        c = pooled[m]
        L.append(f"| {m} | {c['accuracy']:.3f} | {c['precision']:.3f} | {c['recall']:.3f} | {c['f1']:.3f} |")
    L.append("")
    L.append("### 3. 每个 held-out dataset 的表现 (MAE / accuracy)")
    L.append("| fold (held-out) | n | ridge MAE | rf MAE | mlp MAE | rf acc | v6.1 acc |")
    L.append("|---|---|---|---|---|---|---|")
    for held in data["datasets"]:
        m = fm_df[fm_df["fold"] == held]
        def g(model, col):
            return m[m["model"] == model][col].iloc[0]
        L.append(f"| {held} | {int(g('rf','n_test'))} | {g('ridge','mae'):.4f} | {g('rf','mae'):.4f} | "
                 f"{g('mlp','mae'):.4f} | {g('rf','accuracy'):.3f} | {g('v6_1','accuracy'):.3f} |")
    L.append("")
    L.append("### 4. 过拟合检查 (pooled train MAE vs test MAE)")
    L.append("| model | train MAE | test MAE (pooled) | gap |")
    L.append("|---|---|---|---|")
    for m in ["ridge", "rf", "mlp"]:
        L.append(f"| {m} | {train_mae_agg[m]:.4f} | {pooled_reg[m]['mae']:.4f} | "
                 f"{pooled_reg[m]['mae'] - train_mae_agg[m]:+.4f} |")
    L.append("")
    L.append("### 5. MLP 3-seed 稳定性 (per-fold MAE mean±std)")
    L.append("| fold | MAE mean | MAE std | Spearman mean | Spearman std |")
    L.append("|---|---|---|---|---|")
    for s in seed_stats:
        L.append(f"| {s['dataset']} | {s['mae_mean']:.4f} | {s['mae_std']:.4f} | "
                 f"{s['spearman_mean']:.3f} | {s['spearman_std']:.3f} |")
    L.append("")
    L.append("### 6. TEST 未访问")
    L.append("- test_used = false")
    L.append("")
    L.append("## 结论")
    L.append("（仅汇总数据；未修改方法。）")
    L.append("")
    L.append("```")
    L.append('learned_critic_version = "v1"')
    L.append(f"training_samples = {int(len(data['y']))}")
    L.append('evaluation = "leave_one_dataset_out"')
    L.append("forecasting_rerun = false")
    L.append("critic_v6_1_frozen = true")
    L.append("text_embedding_used = false")
    L.append("dataset_id_used_as_feature = false")
    L.append("test_used = false")
    L.append("```")
    (OUT_ROOT / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    main()
