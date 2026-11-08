# Weather follow-up evaluation, January 2021

This package reproduces the January evaluation from saved forecasts and the weather station's measurements. The original study used roof CO₂ concentrations at ten-minute intervals through 1 January 2021 at 00:00. The follow-up uses the same station, measurement unit and sampling interval over the next complete calendar month. The scope of the time holdout is PAER development. The pretraining sources of Chronos-2 have not been resolved to individual stations and months.

From the repository root, run the checker with Python 3 and NumPy.

```sh
python heldout/verify.py
```

The checker extracts 4,464 readings directly from the station's original ZIP and reconstructs all 3,857 contiguous windows. Every window contains 512 context steps followed by 96 measured target steps. The station clock, including the interval end convention of its timestamps, is retained. All target cells in this month contain actual measurements. Adjacent windows share observations.

For each backbone, arm and seed, the checker reselects the original validation weight from its recorded grid, verifies it against the frozen weight, and combines the stored forecasts. It recalculates MSE, MAE, the three-seed means and sample standard deviations, and 2,000 paired moving-block bootstrap replicates using contiguous blocks of 608 prediction origins. The resulting 48 per-seed records, 16 summaries and 24 paired intervals are compared against the published numerical evidence. [Per-seed results](../evidence/weather_2021_01/per_seed.csv) · [Summary](../evidence/weather_2021_01/summary.csv) · [Paired comparisons](../evidence/weather_2021_01/comparisons.csv)

| File | Contents |
|---|---|
| `observations/mpi_roof_2021a.zip` | Original half-year archive. The checker selects the interval from 2021-01-01 00:00 exclusive to 2021-02-01 00:00 inclusive. |
| `observations/labels.npz` | Saved target windows, origins, starts and observed-value mask, used to verify the reconstruction. |
| `predictions/base_*.npz` | Frozen backbone forecasts. Chronos-2 produced identical forecasts across the three seeds and its shared array is stored once. |
| `predictions/memory_*.npz` | Frozen aligned Top-5 memory predictions for retrieval without experts, Raw Expert and PAER, for seeds 42, 123 and 2026. |
| `val_grid.csv`, `frozen_weights.csv` | Original validation MSE/MAE for all candidate weights and the selected weight for each configuration. The Base arm uses its direct forecast. |
| `protocol.json`, `SHA256.json` | Portable record of the bound experiment design, original precommit hash, original array checksums, model replay scope and public input checksums. |

The evaluation plan was recorded before inspecting the follow-up errors. The original time-stamped local record bound 51 artifacts, including training data, model checkpoints and weights. The portable protocol records that record's SHA-256 and the key bindings for published inputs. Original per-model NPY files were packaged using lossless compression without casting their values. The released arrays and validation grids support independent scoring. At the time of the experiment, the frozen backbones reproduced the original validation predictions at 1,024 origins, and the nine memory configurations reproduced the retrieved indices plus original and aligned predictions at 32 validation origins.

The measurements were produced by the Max Planck Institute for Biogeochemistry, Jena, at the Beutenberg roof weather station. The included provider ZIP is unmodified. The derived label file selects and arranges a subset of the readings in forecast windows. Measurements are distributed under CC BY 4.0. [Station archive](https://www.bgc-jena.mpg.de/wetter/weather_data.html) · [Original January–June 2021 ZIP](https://www.bgc-jena.mpg.de/wetter/data/mpi_roof_2021a.zip) · [Station documentation](https://www.bgc-jena.mpg.de/wetter/Weatherstation.pdf) · [License](https://creativecommons.org/licenses/by/4.0/)
