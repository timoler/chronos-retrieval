"""Small deterministic checks of causal windows, aggregation and paired decisions."""
import numpy as np
import pandas as pd

from .data import Series, compile_pool, knowledge_folds
from .gate import decide, merge_context, replace_one
from .refine import decision
from .retrieval import Training, fit, predict, select_weight


def check():
    n = 480
    x = (np.sin(np.arange(n) / 11) + np.arange(n) / 130).astype(np.float32)
    series = Series(x, pd.date_range("2000-01-01", periods=n, freq="h"), 24, 4, 360)
    c, e = knowledge_folds(series.train_end, series.context, series.horizon)[0]
    stage = series.stage(c, e)
    assert stage["bank"][-1] + series.horizon <= stage["fit"][0]
    assert stage["fit"][-1] + series.horizon <= stage["query"][0]
    pool = compile_pool([{"id":"E1","mechanism":"Synthetic low volatility condition",
                          "condition_spec":{"logic":"AND","conditions":[
                              {"feature":"recent_volatility","qualifier":"low"}]}}])
    vector = np.zeros(384, dtype=np.float32)
    vector[0] = 1
    model = fit(stage, pool, lambda _: vector, 42, "cpu", Training(epochs=1))
    result = predict(model, stage, pool)
    idx = result["indices"]
    bank_future = stage["Y"]["bank"].astype(np.float64)
    query_mean = stage["X"]["query"].astype(np.float64).mean(1)
    bank_mean = stage["X"]["bank"].astype(np.float64).mean(1)
    np.testing.assert_array_equal(result["raw"], bank_future[idx].mean(1))
    np.testing.assert_array_equal(result["aligned"], (
        bank_future[idx] + query_mean[:, None, None] - bank_mean[idx][:, :, None]).mean(1))
    assert np.all(stage["bank"][idx] + series.horizon <= stage["query"][:, None])
    assert decision([1,1,1,-0.1], 0.009) == ("UNSUPPORTED",0)
    assert decision([1,1,1,-0.1], 0.01) == ("KEEP",2)
    assert decision([-1,-1,-1,2], 1.0) == ("REJECT",-2)
    assert not decide([4,4,4,4],[4,4,4,4])["accepted"]
    assert decide([4,4,4,4],[3,4,4,4])["accepted"]
    assert select_weight(result["raw"], result["raw"], result["raw"]) == 0
    archive = {"E2":{"id":"E2","mechanism":"Archived"}}
    merged = merge_context(archive, pool)
    updated = replace_one(merged, {"id":"E1","mechanism":"Replacement"})
    assert merged[1] == updated[1] == archive["E2"] and merged[0] == pool[0]
    assert set(model.state_dict()) >= {"logit_gamma","logit_beta","expert_emb_matrix"}
    print("PASS: causal folds, eight-slot training, raw/aligned Top-5, screening and paired gate.")
