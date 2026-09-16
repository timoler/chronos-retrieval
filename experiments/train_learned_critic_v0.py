"""Learned Critic v0 可行性实验 — Leave-One-Dataset-Out (LODO) 验证.

基于已生成的 results/learned_critic_dataset_v1/knowledge_effect_samples.parquet
(48 个 knowledge samples, 5 个数据集), 训练小型 Learned Critic, 做 LODO 验证。

约束(严格遵守):
- 不重新跑 forecasting / LOO 实验, 不重新生成 Expert, 不改 Critic v6.1.
- 回归 target = mean_relative_gain; 辅助标签 = contribution_label (POSITIVE/NEGATIVE/MIXED).
- 特征只用低维通用特征(A/B/C), 禁止 dataset_name/expert_id/真实 label/mean_delta/text embedding.
- 每个 fold 内只对训练集 fit imputer + scaler, 严禁数据泄漏.
- 模型: Ridge / RandomForest / Small MLP (in->32->16->1, ReLU), MLP 3 seeds.
- TEST 全程不访问.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn

# --------------------------------------------------------------------------- #
# config / paths
# --------------------------------------------------------------------------- #
IN_PARQUET = Path("results/learned_critic_dataset_v1/knowledge_effect_samples.parquet")
OUT_ROOT = Path("results/learned_critic_v0")

DATASETS = ["etth1", "electricity", "solar", "weather", "exchange_rate"]
MLP_SEEDS = [42, 123, 2026]
RF_SEED = 42
MLP_EPOCHS = 3000
MLP_LR = 1e-3
MLP_WD = 1e-4

# --------------------------------------------------------------------------- #
# feature definitions
# --------------------------------------------------------------------------- #
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

COVERAGE_MAP = {
    "DEAD": 0.0,
    "CONSTANT_LOW_INFORMATION": 1.0,
    "ACCEPTED_LOCAL_SCOPE": 2.0,
    "OK": 3.0,
}

AUDIT_FEATURES = [
    "v6_1_final_decision",
    "observability_decision",
    "evidence_decision",
    "fidelity_decision",
    "generalization_decision",
    "redundancy_decision",
    "activation_coverage_decision",
    "compiler_decision",
    "repair_happened",
]


def encode_audit(df: pd.DataFrame) -> pd.DataFrame:
    """把 Critic v6.1 audit 字段编码为数值. PASS/ACCEPT/COMPILED/True -> 1."""
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


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def reg_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "spearman": float(spearmanr(y, p)[0]) if len(np.unique(p)) > 1 else float("nan"),
    }


def clf_metrics(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> dict:
    acc = float(accuracy_score(y_true_bin, y_pred_bin))
    p, r, f, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, pos_label=1, zero_division=0, average="binary")
    return {"accuracy": acc, "precision": float(p), "recall": float(r), "f1": float(f)}


def to_binary_pred(pred_cont: np.ndarray) -> np.ndarray:
    """连续预测 -> useful(1)/harmful(0): pred>0 useful, pred<0 harmful."""
    return (pred_cont > 0).astype(int)


# --------------------------------------------------------------------------- #
# small MLP
# --------------------------------------------------------------------------- #
class MLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

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
    final_loss = float("nan")
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = lossfn(model(Xtr_t), ytr_t)
        loss.backward()
        opt.step()
        final_loss = float(loss.item())
    model.eval()
    with torch.no_grad():
        train_pred = model(Xtr_t).numpy()
        test_pred = model(Xte_t).numpy()
    return train_pred, test_pred, final_loss


# --------------------------------------------------------------------------- #
# data prep
# --------------------------------------------------------------------------- #
def load_data() -> dict:
    df = pd.read_parquet(IN_PARQUET)
    audit = encode_audit(df)
    feature_cols = GENERIC_FEATURES + ACTIVATION_FEATURES + AUDIT_FEATURES
    X = pd.DataFrame(index=df.index)
    for c in GENERIC_FEATURES:
        X[c] = df[c].astype(float)
    for c in ACTIVATION_FEATURES:
        X[c] = df[c].astype(float)
    for c in AUDIT_FEATURES:
        X[c] = audit[c].astype(float)
    X = X[feature_cols].to_numpy(dtype=float)

    return {
        "X": X,
        "y": df["mean_relative_gain"].to_numpy(dtype=float),
        "label": df["contribution_label"].to_numpy(),
        "dataset": df["dataset_name"].to_numpy(),
        "expert_id": df["expert_id"].to_numpy(),
        "expert_text": df["expert_text"].to_numpy(),
        "v6_1_decision": df["v6_1_final_decision"].to_numpy(),  # baseline B
        "feature_cols": feature_cols,
    }


def build_folds(data: dict):
    """5 个 LODO fold, 每个 fold 的 held-out dataset 完全不进入训练."""
    folds = []
    for held in DATASETS:
        te_idx = np.where(data["dataset"] == held)[0]
        tr_idx = np.where(data["dataset"] != held)[0]
        folds.append((held, tr_idx, te_idx))
    return folds


# --------------------------------------------------------------------------- #
# per-fold preprocessing (imputer + scaler fit on TRAIN only)
# --------------------------------------------------------------------------- #
def preprocess_fold(Xtr, Xte):
    # drop near-constant columns (std < 1e-12) using TRAIN only
    tr_std = Xtr.std(axis=0)
    keep = tr_std > 1e-12
    Xtr_k = Xtr[:, keep]
    Xte_k = Xte[:, keep]

    imp = SimpleImputer(strategy="median")
    Xtr_i = imp.fit_transform(Xtr_k)
    Xte_i = imp.transform(Xte_k)

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr_i)
    Xte_s = scaler.transform(Xte_i)
    return Xtr_s, Xte_s, int(keep.sum())


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    data = load_data()
    folds = build_folds(data)
    print(f"training samples = {len(data['y'])} | features = {data['X'].shape[1]} | "
          f"folds = {len(folds)} | device = cpu", flush=True)

    rows = []           # per-sample predictions
    fold_metrics = []   # per-fold per-model metrics
    fold_mlp_seed_stats = []  # MLP per-seed stability per fold
    fold_dropped = {}   # per-fold feature count info

    for held, tr_idx, te_idx in folds:
        Xtr_raw, Xte_raw = data["X"][tr_idx], data["X"][te_idx]
        ytr, yte = data["y"][tr_idx], data["y"][te_idx]
        Xtr, Xte, n_feat = preprocess_fold(Xtr_raw, Xte_raw)
        fold_dropped[held] = {"raw_features": int(Xtr_raw.shape[1]), "used_features": n_feat}

        # ---- Ridge ----
        ridge = Ridge(alpha=1.0)
        ridge.fit(Xtr, ytr)
        ridge_tr = ridge.predict(Xtr)
        ridge_te = ridge.predict(Xte)

        # ---- RandomForest ----
        rf = RandomForestRegressor(n_estimators=500, random_state=RF_SEED)
        rf.fit(Xtr, ytr)
        rf_tr = rf.predict(Xtr)
        rf_te = rf.predict(Xte)

        # ---- MLP (3 seeds) ----
        mlp_te_seeds, mlp_tr_seeds = [], []
        for s in MLP_SEEDS:
            _, te_pred, _ = train_mlp(Xtr, ytr, Xte, seed=s)
            tr_pred = None
            mlp_te_seeds.append(te_pred)
            _, tr_pred, _ = train_mlp(Xtr, ytr, Xtr, seed=s)
            mlp_tr_seeds.append(tr_pred)
        mlp_te = np.mean(mlp_te_seeds, axis=0)
        mlp_te_std = np.std(mlp_te_seeds, axis=0)
        mlp_tr = np.mean(mlp_tr_seeds, axis=0)

        # per-seed regression metrics (stability)
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

        # record per-sample predictions
        te_ids = np.where(data["dataset"] == held)[0]
        for k, idx in enumerate(te_ids):
            rows.append({
                "fold": held,
                "dataset_name": data["dataset"][idx],
                "expert_id": data["expert_id"][idx],
                "expert_text": data["expert_text"][idx],
                "true_label": data["label"][idx],
                "true_mean_relative_gain": float(yte[k]),
                "ridge_pred": float(ridge_te[k]),
                "rf_pred": float(rf_te[k]),
                "mlp_pred": float(mlp_te[k]),
                "mlp_pred_std": float(mlp_te_std[k]),
                "mlp_pred_seed42": float(mlp_te_seeds[0][k]),
                "mlp_pred_seed123": float(mlp_te_seeds[1][k]),
                "mlp_pred_seed2026": float(mlp_te_seeds[2][k]),
                "v6_1_decision": data["v6_1_decision"][idx],
            })

        # ---- fold metrics ----
        # regression (all samples incl MIXED)
        regs = {
            "ridge": reg_metrics(yte, ridge_te),
            "rf": reg_metrics(yte, rf_te),
            "mlp": reg_metrics(yte, mlp_te),
        }
        # classification on POSITIVE/NEGATIVE subset only
        mask = (data["label"][te_idx] != "MIXED")
        y_true_bin = (data["label"][te_idx][mask] == "POSITIVE").astype(int)
        clfs = {}
        for mname, pte in [("ridge", ridge_te), ("rf", rf_te), ("mlp", mlp_te)]:
            clfs[mname] = clf_metrics(y_true_bin, to_binary_pred(pte[mask]))
        # baselines
        clfs["always_positive"] = clf_metrics(y_true_bin, np.ones_like(y_true_bin))
        v6_bin = (data["v6_1_decision"][te_idx][mask] == "ACCEPT").astype(int)
        clfs["v6_1"] = clf_metrics(y_true_bin, v6_bin)

        for mname in ["ridge", "rf", "mlp"]:
            fold_metrics.append({
                "fold": held, "model": mname, "n_test": int(len(yte)), "n_class": int(mask.sum()),
                **regs[mname], **clfs[mname],
            })
        for bname in ["always_positive", "v6_1"]:
            fold_metrics.append({
                "fold": held, "model": bname, "n_test": int(len(yte)), "n_class": int(mask.sum()),
                "mae": float("nan"), "rmse": float("nan"), "spearman": float("nan"), **clfs[bname],
            })

        print(f"  [fold {held}] n_test={len(yte)} n_feat={n_feat} | "
              f"ridge MAE={regs['ridge']['mae']:.4f} rf MAE={regs['rf']['mae']:.4f} "
              f"mlp MAE={regs['mlp']['mae']:.4f} | "
              f"acc ridge={clfs['ridge']['accuracy']:.3f} rf={clfs['rf']['accuracy']:.3f} "
              f"mlp={clfs['mlp']['accuracy']:.3f} v6.1={clfs['v6_1']['accuracy']:.3f}", flush=True)

    # ------------------------------------------------------------------ #
    # aggregate
    # ------------------------------------------------------------------ #
    pred_df = pd.DataFrame(rows)
    fm_df = pd.DataFrame(fold_metrics)

    # pooled classification (POS/NEG only)
    pooled = {}
    for mname in ["ridge", "rf", "mlp"]:
        sub = pred_df[pred_df["true_label"] != "MIXED"]
        y_true_bin = (sub["true_label"] == "POSITIVE").astype(int).to_numpy()
        y_pred_bin = to_binary_pred(sub[f"{mname}_pred"].to_numpy())
        pooled[mname] = clf_metrics(y_true_bin, y_pred_bin)
    sub = pred_df[pred_df["true_label"] != "MIXED"]
    y_true_bin = (sub["true_label"] == "POSITIVE").astype(int).to_numpy()
    pooled["always_positive"] = clf_metrics(y_true_bin, np.ones_like(y_true_bin))
    pooled["v6_1"] = clf_metrics(y_true_bin, (sub["v6_1_decision"] == "ACCEPT").astype(int).to_numpy())

    # pooled regression (all samples)
    pooled_reg = {}
    for mname in ["ridge", "rf", "mlp"]:
        pooled_reg[mname] = reg_metrics(
            pred_df["true_mean_relative_gain"].to_numpy(),
            pred_df[f"{mname}_pred"].to_numpy())

    # regression mean±std across folds
    agg = {}
    for mname in ["ridge", "rf", "mlp"]:
        m = fm_df[fm_df["model"] == mname]
        agg[mname] = {
            "mae_mean": float(m["mae"].mean()), "mae_std": float(m["mae"].std()),
            "rmse_mean": float(m["rmse"].mean()), "rmse_std": float(m["rmse"].std()),
            "spearman_mean": float(m["spearman"].mean()), "spearman_std": float(m["spearman"].std()),
        }

    # train MAE (overfitting check): pooled train MAE per model
    train_mae = {}
    for held, tr_idx, te_idx in folds:
        Xtr_raw, Xte_raw = data["X"][tr_idx], data["X"][te_idx]
        ytr, yte = data["y"][tr_idx], data["y"][te_idx]
        Xtr, Xte, n_feat = preprocess_fold(Xtr_raw, Xte_raw)
        ridge = Ridge(alpha=1.0).fit(Xtr, ytr)
        rf = RandomForestRegressor(n_estimators=500, random_state=RF_SEED).fit(Xtr, ytr)
        mlp_tr = np.mean([train_mlp(Xtr, ytr, Xtr, seed=s)[1] for s in MLP_SEEDS], axis=0)
        for mname, tr_pred in [("ridge", ridge.predict(Xtr)), ("rf", rf.predict(Xtr)), ("mlp", mlp_tr)]:
            train_mae.setdefault(mname, []).append(mean_absolute_error(ytr, tr_pred))
    train_mae_agg = {m: float(np.mean(v)) for m, v in train_mae.items()}

    # ------------------------------------------------------------------ #
    # write outputs
    # ------------------------------------------------------------------ #
    pred_df.to_csv(OUT_ROOT / "lodo_predictions.csv", index=False, encoding="utf-8")
    fm_df.to_csv(OUT_ROOT / "lodo_metrics.csv", index=False, encoding="utf-8")

    # model_comparison.csv
    comp_rows = []
    for mname in ["ridge", "rf", "mlp"]:
        comp_rows.append({
            "model": mname,
            "mae_mean": agg[mname]["mae_mean"], "mae_std": agg[mname]["mae_std"],
            "rmse_mean": agg[mname]["rmse_mean"], "rmse_std": agg[mname]["rmse_std"],
            "spearman_mean": agg[mname]["spearman_mean"], "spearman_std": agg[mname]["spearman_std"],
            "pooled_mae": pooled_reg[mname]["mae"], "pooled_rmse": pooled_reg[mname]["rmse"],
            "pooled_spearman": pooled_reg[mname]["spearman"],
            "pooled_accuracy": pooled[mname]["accuracy"], "pooled_f1": pooled[mname]["f1"],
            "pooled_precision": pooled[mname]["precision"], "pooled_recall": pooled[mname]["recall"],
            "train_mae": train_mae_agg[mname],
        })
    for bname in ["always_positive", "v6_1"]:
        comp_rows.append({
            "model": bname,
            "mae_mean": float("nan"), "mae_std": float("nan"),
            "rmse_mean": float("nan"), "rmse_std": float("nan"),
            "spearman_mean": float("nan"), "spearman_std": float("nan"),
            "pooled_mae": float("nan"), "pooled_rmse": float("nan"),
            "pooled_spearman": float("nan"),
            "pooled_accuracy": pooled[bname]["accuracy"], "pooled_f1": pooled[bname]["f1"],
            "pooled_precision": pooled[bname]["precision"], "pooled_recall": pooled[bname]["recall"],
            "train_mae": float("nan"),
        })
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(OUT_ROOT / "model_comparison.csv", index=False, encoding="utf-8")

    # exchange holdout predictions
    exch = pred_df[pred_df["dataset_name"] == "exchange_rate"][
        ["expert_id", "expert_text", "true_mean_relative_gain", "true_label",
         "ridge_pred", "rf_pred", "mlp_pred", "mlp_pred_std"]].copy()
    exch.to_csv(OUT_ROOT / "exchange_holdout_predictions.csv", index=False, encoding="utf-8")

    # feature_list.json
    feature_list = {
        "target": "mean_relative_gain",
        "label": "contribution_label (POSITIVE/NEGATIVE/MIXED)",
        "n_training_samples": int(len(data["y"])),
        "evaluation": "leave_one_dataset_out",
        "forbidden_features": [
            "dataset_name", "expert_id", "contribution_label", "mean_delta",
            "full_raw_mse_*", "without_i_mse_*", "delta_seed*", "relative_gain_seed*",
            "text embedding (384-d MiniLM)", "TEST set",
        ],
        "features": {
            "A_dataset_generic": GENERIC_FEATURES,
            "B_expert_activation": ACTIVATION_FEATURES,
            "C_critic_audit": AUDIT_FEATURES,
        },
        "audit_encoding": {
            "v6_1_final_decision": {"ACCEPT": 1, "REJECT": 0},
            "observability/evidence/fidelity/generalization": {"PASS": 1, "FAIL": 0},
            "redundancy_decision": {"DUPLICATE_ACTIVATION": 1, "else": 0},
            "activation_coverage_decision": COVERAGE_MAP,
            "compiler_decision": {"COMPILED": 1, "else": 0},
            "repair_happened": {"True": 1, "False": 0},
        },
        "n_raw_features": int(data["X"].shape[1]),
    }
    (OUT_ROOT / "feature_list.json").write_text(
        json.dumps(feature_list, ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ #
    # report.md
    # ------------------------------------------------------------------ #
    L = []
    L.append("# Learned Critic v0 — LODO 验证报告\n")

    L.append("## 结果\n")

    L.append("### 1. 每个模型 5-fold LODO 回归指标 (MAE / RMSE / Spearman, mean±std across 5 folds)\n")
    L.append("| model | MAE (mean±std) | RMSE (mean±std) | Spearman (mean±std) |")
    L.append("|---|---|---|---|")
    for mname in ["ridge", "rf", "mlp"]:
        a = agg[mname]
        L.append(f"| {mname} | {a['mae_mean']:.4f} ± {a['mae_std']:.4f} | "
                 f"{a['rmse_mean']:.4f} ± {a['rmse_std']:.4f} | "
                 f"{a['spearman_mean']:.3f} ± {a['spearman_std']:.3f} |")
    L.append("")

    L.append("### 2. Positive/Negative 二分类 (pooled over 5 folds, MIXED 不参与)\n")
    L.append("| model | accuracy | precision | recall | F1 |")
    L.append("|---|---|---|---|---|")
    for mname in ["ridge", "rf", "mlp", "always_positive", "v6_1"]:
        c = pooled[mname]
        L.append(f"| {mname} | {c['accuracy']:.3f} | {c['precision']:.3f} | {c['recall']:.3f} | {c['f1']:.3f} |")
    L.append("")

    L.append("### 3. 每个 held-out dataset 的表现 (MAE / accuracy)\n")
    L.append("| fold (held-out) | n_test | ridge MAE | rf MAE | mlp MAE | ridge acc | rf acc | mlp acc | v6.1 acc |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for held in DATASETS:
        m = fm_df[fm_df["fold"] == held]
        def g(model, col):
            return m[m["model"] == model][col].iloc[0]
        L.append(f"| {held} | {int(g('ridge','n_test'))} | {g('ridge','mae'):.4f} | {g('rf','mae'):.4f} | "
                 f"{g('mlp','mae'):.4f} | {g('ridge','accuracy'):.3f} | {g('rf','accuracy'):.3f} | "
                 f"{g('mlp','accuracy'):.3f} | {g('v6_1','accuracy'):.3f} |")
    L.append("")

    L.append("### 4. Exchange held-out 详细预测\n")
    L.append("| expert_id | true gain | true label | ridge | rf | mlp |")
    L.append("|---|---|---|---|---|---|")
    for _, r in exch.iterrows():
        L.append(f"| {r['expert_id']} | {r['true_mean_relative_gain']:.5f} | {r['true_label']} | "
                 f"{r['ridge_pred']:.5f} | {r['rf_pred']:.5f} | {r['mlp_pred']:.5f} |")
    L.append("")

    L.append("### 5. v6.1 baseline 表现 (pooled classification)\n")
    c = pooled["v6_1"]
    L.append(f"- accuracy={c['accuracy']:.3f}, precision={c['precision']:.3f}, "
             f"recall={c['recall']:.3f}, F1={c['f1']:.3f}\n")

    L.append("### 6. Always Positive baseline 表现 (pooled classification)\n")
    c = pooled["always_positive"]
    L.append(f"- accuracy={c['accuracy']:.3f}, precision={c['precision']:.3f}, "
             f"recall={c['recall']:.3f}, F1={c['f1']:.3f}\n")

    L.append("### 7. MLP 3 seeds 稳定性 (per-fold MAE mean±std over 3 seeds)\n")
    L.append("| fold | MAE mean | MAE std | RMSE mean | RMSE std | Spearman mean | Spearman std |")
    L.append("|---|---|---|---|---|---|---|")
    for s in fold_mlp_seed_stats:
        L.append(f"| {s['dataset']} | {s['mae_mean']:.4f} | {s['mae_std']:.4f} | "
                 f"{s['rmse_mean']:.4f} | {s['rmse_std']:.4f} | "
                 f"{s['spearman_mean']:.3f} | {s['spearman_std']:.3f} |")
    L.append("")

    L.append("### 8. 过拟合检查 (pooled train MAE vs test MAE)\n")
    L.append("| model | train MAE | test MAE (pooled) | gap |")
    L.append("|---|---|---|---|")
    for mname in ["ridge", "rf", "mlp"]:
        tr = train_mae_agg[mname]
        te = pooled_reg[mname]["mae"]
        L.append(f"| {mname} | {tr:.4f} | {te:.4f} | {te-tr:+.4f} |")
    L.append("")

    L.append("### 9. TEST 未访问\n")
    L.append("- test_used = false (本实验只读 knowledge_effect_samples.parquet, 未触碰任何 TEST split)\n")

    L.append("## 结论\n")

    # A/B/C 判定 (基于数据事实, 不做超纲评价)
    best_f1 = max((pooled[m]["f1"], m) for m in ["ridge", "rf", "mlp"])
    v61_f1 = pooled["v6_1"]["f1"]
    ap_acc = pooled["always_positive"]["accuracy"]
    # exchange: 模型是否把 exchange 样本预测为负(<0)
    exch_neg_frac = float((exch[["ridge_pred", "rf_pred", "mlp_pred"]] < 0).mean().mean())
    exch_true_neg_frac = float((exch["true_mean_relative_gain"] < 0).mean())

    L.append(f"- Learned Critic 最强模型 = {best_f1[1]} (pooled F1={best_f1[0]:.3f}); "
             f"v6.1 pooled F1={v61_f1:.3f}.")
    L.append(f"- Always Positive baseline accuracy={ap_acc:.3f} (POS/NEG 子集 30/7 严重不平衡, "
             f"多数类 baseline 本身很高).")
    L.append(f"- Exchange held-out: 模型平均预测为负的比例 = {exch_neg_frac:.3f}, "
             f"真实负(gain<0)比例 = {exch_true_neg_frac:.3f}.")
    L.append("")
    L.append("- A. Learned Critic 是否比 v6.1 更能预测真实 contribution: 见上表 F1 对比 "
             f"(best {best_f1[0]:.3f} vs v6.1 {v61_f1:.3f}).")
    L.append("- B. 未见过数据集上的可迁移信号: 以 Spearman(mean±std across folds) 与 Exchange "
             "holdout 预测符号判断.")
    L.append("- C. 是否值得扩展到 10–15 数据集: 依据上面 A/B 的结果, 见最终判定.")
    L.append("")
    L.append("```")
    L.append('learned_critic_version = "v0"')
    L.append(f"training_samples = {int(len(data['y']))}")
    L.append('evaluation = "leave_one_dataset_out"')
    L.append("forecasting_rerun = false")
    L.append("critic_v6_1_frozen = true")
    L.append("text_embedding_used = false")
    L.append("dataset_id_used_as_feature = false")
    L.append("test_used = false")
    L.append("```")

    (OUT_ROOT / "report.md").write_text("\n".join(L), encoding="utf-8")

    # final print block
    print("\n===== FINAL =====", flush=True)
    print('learned_critic_version = "v0"', flush=True)
    print(f"training_samples = {int(len(data['y']))}", flush=True)
    print('evaluation = "leave_one_dataset_out"', flush=True)
    print("forecasting_rerun = false", flush=True)
    print("critic_v6_1_frozen = true", flush=True)
    print("text_embedding_used = false", flush=True)
    print("dataset_id_used_as_feature = false", flush=True)
    print("test_used = false", flush=True)

    print("\n===== MODEL COMPARISON (pooled) =====", flush=True)
    print(comp_df[["model", "pooled_mae", "pooled_rmse", "pooled_spearman",
                   "pooled_accuracy", "pooled_f1"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
