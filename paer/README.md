# PAER numerical method

Run the commands from the repository root. Install NumPy, pandas and PyTorch in the same environment. A CUDA build and the original numerical environment are needed to compare individual model outputs with the archived CUDA runs. The protocol check runs on CPU.

```bash
python -m paer check
python -m paer evaluate --series inputs/weather/series.csv --protocol inputs/weather/protocol.json --pool inputs/weather/initial.json --embeddings inputs/embeddings --out runs/weather-contributions
python -m paer refine --series inputs/weather/series.csv --protocol inputs/weather/protocol.json --pool inputs/weather/initial.json --proposals inputs/weather/proposals.json --embeddings inputs/embeddings --out runs/weather-refinement
python -m paer deploy --series inputs/weather/series.csv --protocol inputs/weather/protocol.json --pool inputs/weather/experts.json --embeddings inputs/embeddings --seed 42 --out runs/weather-memory
```

Each command takes a new or empty output directory. The built-in check fits a synthetic 1-epoch model and checks time order, Top-5 alignment, the coverage override and the paired gate. The research commands fit 30 epochs with AdamW, an eight-slot encoder, 256 sampled candidates per minibatch, batch size 64, gradient clipping at 5 and temperature 1. Contributions and revision gates use four expanding folds with seed 42. Each fold refits the feature scaler from its historical bank.

`series.csv` contains chronological, preprocessed `date,OT` observations. `protocol.json` supplies integer `context` and `horizon` lengths, plus `split.train`, `split.val` and `split.test` as contiguous half-open row ranges. Ordinal timestamps can be marked with `"calendar": "disabled"`. Supplied exchange-rate protocols use this restriction.

The initial and final pool files contain arrays of records with an `id` in `E1` through `E8`, a `mechanism` string and a `condition_spec` object. Each spec uses `"logic": "AND"` or `"OR"` and a list of `{"feature": ..., "qualifier": ...}` conditions. The compiler rebuilds the executable condition for each record. Stored unit-length embeddings use the archived naming convention, SHA-256 of the UTF-8 JSON-serialized mechanism string with `ensure_ascii=False`, followed by `.npy`. With an explicit `--encoder` path, sentence-transformers can encode cache misses using the local all-MiniLM-L6-v2 snapshot.

`evaluate` retrains the complete pool and each single-slot mask with the same seed. Each per-fold reward is the masked-minus-full raw Top-5 memory MSE, divided by the masked MSE. The first matching screening rule determines KEEP, REJECT, REVISE or UNSUPPORTED. The output records full and masked MSE under both raw and mean-aligned aggregation.

`refine` takes the saved revision proposals for reproducible numerical replay. They are a JSON object keyed by one-based round numbers (`"1"`, `"2"`). Each executed round must supply the pending expert IDs from that round's observed decisions. Each proposed replacement competes with its incumbent with all other experts fixed. A strict decrease in four-fold mean raw MSE accepts it. Archived KEEP versions exit the active pool. After at most two revision rounds the method returns the archived versions. A different numerical environment can lead to different decisions and a mismatch with the recorded proposal IDs. The Python API `paer.refine.refine(initial, proposal_provider, evaluator, calendar=True)` also accepts a provider function taking the round number and evaluation records.

`deploy` fits one final retriever on TRAIN. It applies the same learned weights to the held-out VAL and TEST windows and saves `retriever.pt`, `val.npz`, `test.npz` and `metrics.json`. The archives contain window starts, top candidate indices, raw and aligned memory predictions, and labels. Use seeds 42, 123 and 2026 in separate runs for the three-seed comparison.

To fuse a particular base model, add `--base-val BASE_VAL.npz --base-test BASE_TEST.npz` to the deployment command. Each file must contain matching absolute context-start row indices under `starts` and predictions in original target units with shape `(number_of_windows, horizon)` under `prediction`. The command saves the grid weight minimizing VAL MSE, breaks exact ties in favor of the smaller weight and applies that fixed weight to TEST.
