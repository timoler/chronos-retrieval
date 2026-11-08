# Experiment evidence

Run `python tools/verify_results.py` from the repository root to recompute the published means, sample standard deviations, paired improvements, revision acceptance counts, and external-period validation-weight reuse. Append `--json` for machine-readable results. The verifier uses the Python standard library and the tables in this directory.

## Seven single-target tasks

[Per-seed scores](main/per_seed.csv) cover seven tasks, four backbones, three arms, and three retriever seeds (42, 123, 2026), for 252 records. [Summaries](main/summary.csv) report metrics in the original units. MSE selects the fusion weight on the original validation split. Larger positive RI indicates a larger reduction in error. For each task and backbone, RI is calculated from seed means as `100 * (reference - arm) / reference`. Cross-task RI gives equal weight to each task–backbone combination.

| Arm | Knowledge used by the frozen retrieval branch |
|---|---|
| Raw Expert | Eight initial experts with their executable conditions |
| Revision without acceptance | Contribution screening and language-model revision. Each proposed version enters the next screening round. |
| PAER | Contribution screening and a paired comparison of the incumbent and proposed version within a common knowledge pool |

PAER reduces mean MSE relative to the backbone on 25 of 28 task–backbone combinations, ties on one, and increases MSE on two. Relative to Raw Expert the counts are 13, 1, and 14. Relative to revision without acceptance they are 20, 0, and 8. The original test periods were available for development diagnostics. [Paired comparisons](main/comparisons.csv) · [Memory-only scores](main/memory_per_seed.csv) · [Revision decisions](main/revision_acceptance.csv)

The four feedback folds expand in chronological order. Candidate labels end before the first fitting origin, fitting labels end before the first evaluation origin, and feature statistics use each fold's candidate contexts. The [block interval tables](main/block_intervals/) summarize paired differences from 2,000 resamples of consecutive forecast origins. The primary block length is context length plus horizon. Interval estimation conditions on the frozen predictions. The standard-library verifier recomputes summary statistics from per-seed errors and verifies the published interval table checksums. Re-estimating interval endpoints would require per-origin losses.

## Five-task component controls

[The component controls](controls/per_seed.csv) cover ETTh1, ETTh2, Exchange Rate, Traffic, and Weather with eight configurations, four backbones and three seeds. The 480 records use the same numerical engine, input bindings, frozen backbone predictions, and validation weight protocol as the main comparison. The verifier checks the overlapping Raw Expert and Revision without acceptance rows against the main study. Solar and Electricity have main-study results and lack this complete set of controls.

Numeric retrieval uses the learned numerical score. Retrieval without experts retains the learned numerical and residual-shape scores with zero expert inputs. Screening only uses the positive decisions from initial screening. The unconditional-expert and shuffled-text controls use the final expert pool obtained by revision without the additional acceptance comparison. These controls evaluate that pool. Across the five tasks PAER has lower MSE than retrieval without experts in 14 of 20 combinations, and lower MSE than initial screening alone in 8 of 20, with eight ties.

## Weather in January 2021

The next complete month of measurements from the same Jena rooftop CO₂ sensor contains 4,464 consecutive ten-minute observations and 3,857 eligible windows of length 512 plus horizon 96. The four configurations use the original checkpoints and validation-selected weights. January contributes no new missing observations. Every reported seed and backbone uses the same windows. [Dataset receipt](weather_2021_01/data_source.json) · [Per-seed scores](weather_2021_01/per_seed.csv) · [Summaries](weather_2021_01/summary.csv) · [Paired differences and intervals](weather_2021_01/comparisons.csv)

| Backbone | Base MSE | Retrieval without experts MSE | Raw Expert MSE | PAER MSE |
|---|---:|---:|---:|---:|
| Chronos-2 | 49.2521 | 60.7999 | 50.0090 | 51.8783 |
| PatchTST | 60.7924 | 64.4820 | 60.7924 | 60.7924 |
| iTransformer | 60.7260 | 62.0762 | 58.5958 | 58.7205 |
| RAFT | 59.9138 | 61.6091 | 58.7333 | 58.8853 |

The period postdates all Weather observations used in PAER development and was specified before this evaluation. Foundation-model pretraining coverage of this station and month is unknown. The scientific data source is Max Planck Institute for Biogeochemistry, Jena, Weather Station Beutenberg, under CC BY 4.0. [Source archive](https://www.bgc-jena.mpg.de/wetter/weather_data.html) · [License](https://creativecommons.org/licenses/by/4.0/)

## Dataset and result provenance

[Dataset metadata](datasets.json) gives the target definition, sampling rate, window settings, exact row boundaries, window counts, and hashes of the bound series and window indices. The ETT tasks use transformer oil temperature. Exchange Rate uses the seventh series with calendar conditions disabled. Traffic uses occupancy for Sensor_000. Solar sums 137 channels. Electricity sums all 370 original load series and retains their recorded zeros. Weather uses the same rooftop sensor for the original and subsequent periods. [ETT](https://github.com/zhouhaoyi/ETDataset) · [Exchange Rate, Traffic and Solar benchmark data](https://github.com/laiguokun/multivariate-time-series-data) · [UCI ElectricityLoadDiagrams20112014](https://doi.org/10.24432/C58C86) · [MPI-BGC Weather](https://www.bgc-jena.mpg.de/wetter/weather_data.html)

[Source checksums](sources.json) retain the SHA-256 of each original result table and its released numerical projection. Public tables preserve the recorded numeric fields, identify each arm by experimental function, and omit machine-specific job locations. Run-level hashes identify the seven-task protocol, the external-period evaluation plan, the data receipt and the independent numerical audit.
