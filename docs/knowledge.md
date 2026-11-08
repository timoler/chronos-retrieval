# Expert revision from training evidence

Each task starts with eight recorded hypotheses in `inputs/<task>/initial.json`. Their mechanisms describe patterns to inspect. Their conditions refer to features available at a forecast origin. The retriever separately computes the activation of each condition for its query and candidate windows.

The program evaluates an active pool and each leave-one-out intervention with the same seed and four expanding TRAIN folds. It calculates the fold contributions from the MSE of the raw, equally weighted top-five futures. A slot's activation coverage is averaged over the four folds. The contribution verdict and reward are fixed by the program before any language-model call.

| Stage | Information supplied | Output |
|---|---|---|
| Critic | The pending expert, raw contribution statistics, coverage, fixed verdict, integer reward and active pool size | Expert ID, unchanged verdict, diagnosis and revision instruction |
| Expert | Observable catalog, executable condition schema, the same TRAIN records including aligned-memory diagnostics, and the Critic response | One proposed version per pending ID, with a replacement flag |
| Acceptance | The old and proposed version in the same archived-plus-active pool | Accept when mean four-fold raw memory MSE strictly decreases |

KEEP versions go directly to the archive. REVISE, REJECT and UNSUPPORTED versions can receive proposals within the remaining two revision rounds. The aligned diagnostic compares two versions of the TRAIN memory forecast. The statistical evidence supplied to the prompts ends at the TRAIN decision folds. Deployment validation selects the fusion weight later.

## Prepare requests and replay responses

Install the numerical dependencies specified for the main package. The following commands read local files. Supply responses generated for the displayed requests. The request JSON has `model`, `system`, `messages` and `max_tokens` fields. The recorded model identifier is `deepseek-v4-pro`, the output cap is 20,000 tokens, and the original requests omitted LLM temperature and seed.

First obtain `evaluation.json` from `python -m paer evaluate` for the current active pool. Pass that file and the next one-based revision number to the prompt tool. This example uses the initial Weather pool.

```sh
python tools/prepare_revision.py --dataset weather --round 1 --evaluation run/evaluation.json --out feedback/r1
```

`feedback/r1/critic_request.json` contains the complete Critic prompt and evidence. Save the response as a JSON object with a `critiques` array. Every entry has `expert_id`, `verdict`, `diagnosis` and `revision_instruction`. The tool verifies complete ID coverage and preserves the numerical verdict. The next command writes `expert_request.json` with the complete Expert prompt.

```sh
python tools/prepare_revision.py --dataset weather --round 1 --evaluation run/evaluation.json --critic-reply critic.json --out feedback/r1
```

Save the Expert response as a JSON object with an `experts` array. Each version preserves the original ID, updates the mechanism and executable condition, and supplies a Boolean `replaced` flag. The parser also accepts a local response envelope containing the returned JSON in a `text` field. The final command compiles the proposed conditions, associates each version with its lineage and writes `proposals.json`.

```sh
python tools/prepare_revision.py --dataset weather --round 1 --evaluation run/evaluation.json --critic-reply critic.json --expert-reply expert.json --out feedback/r1
```

The proposal file maps the revision number to a list of expert objects. The checked-in `inputs/<task>/proposals.json` uses this format for all recorded rounds and can be passed to `python -m paer refine --proposals` with the task inputs. For new responses, collect the next round's active-pool evaluation after the current round's screening and acceptance decisions, then combine the per-round proposal mappings. New mechanism texts need embeddings produced by the same frozen encoder. API callers can supply a callable `proposal_rounds(round_number, records)` to `paer.refine.refine` to obtain proposals between rounds.

The validator checks a nonempty mechanism and compiles flat AND/OR conditions against the runtime feature catalog. Continuous features use high or low. Calendar predicates require a catalog category. Exchange Rate uses ordinal dates and forbids calendar conditions. Traffic and Solar use reconstructed calendar positions. A failed schema check raises an error. The recorded workflow allowed up to two same-provider repairs driven by validation errors. This offline interface writes prompt payloads and consumes existing replies. Running it again with new online responses may produce different proposals.

The release includes the accepted input formats, recorded proposal artifacts and frozen numerical results. The original initial-generation records and some historical transport records are incomplete. Verification of the offline bridge included a recorded Weather second-round pair. The reconstructed Critic and Expert system and user messages matched their saved messages, and the resulting mechanism and condition matched the published proposal.
