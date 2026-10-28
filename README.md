# The language model on a drifting speech decoder

Code, results and preregistered predictions for a study of the language-model rescoring stage of an
intracortical speech neuroprosthesis, evaluated on sessions recorded after the decoder's calibration.
Data: Card et al. (NEJM 2024), participant T15, 45 sessions over 20 months, and participant T12 (Willett et
al., Nature 2023) for replication. The work builds on Gabriele Cinà's cached candidate lists and rescorer
scores ([how-much-language-model](https://github.com/gabrycina/how-much-language-model),
[arXiv:2609.33538](https://arxiv.org/abs/2609.33538)).

Preprint: `paper/rescoring-drift.pdf` · [doi:10.5281/zenodo.23745489](https://doi.org/10.5281/zenodo.23745489) · Data: [doi:10.5281/zenodo.23745520](https://doi.org/10.5281/zenodo.23745520)

```sh
pip install scipy
python3 reproduce.py
```

`reproduce.py` makes 349 checks, one for each number in the paper that comes from this work's results,
recomputing it from the committed files in `results/` and `derived/step4/`. It runs on a CPU in seconds and
exits non-zero on any mismatch. Expected output ends with `349/349 checks passed`.

## Findings

1. **On a calibrated decoder, rescoring is churn with a small net.** On Cinà's random test split
   Qwen2.5-7B fixes 85 words and breaks 63. Held out by session, OPT-6.7b nets exactly zero on T12 (33
   sentences helped, 30 harmed). A perfect correct-or-keep gate would gain 0.27 to 2.0 points, more than
   any rescorer's net gain.
2. **An output-feature gate halves harm on T15 and replicates on T12 for OPT only.** On T15's later
   sessions it cuts Qwen's harmful changes from 34 to 17 at no word-error cost; the registered T12
   replication (H1) fails, holding for OPT alone.
3. **The rescorer's activations add nothing.** A probe on OPT-6.7b's residual stream makes the gate worse
   by 0.11 points (95% interval +0.04 to +0.20).
4. **Drift is immediate.** A decoder trained on the first 21 sessions has 3× the recalibrated decoder's
   phoneme error rate one week later, 36% by day 21, 64% by day 103, and 74–81% after a year.
5. **Under drift the rescorer matters more, and the gate stops paying.** Within a month of calibration
   the rescorer removes a larger share of errors and the gate vetoes helpful changes. After three months
   the 100-best lists hold the reference for 6 of 462 sentences; the fix is recalibration.

## Preregistered verdicts

Every prediction was committed before the analysis that tests it, and from step 3a on a copy was stamped
with OpenTimestamps (`timestamps/`). Full text and outcomes: `PREDICTIONS.md`.

| Step | Prediction | Verdict |
|---|---|---|
| 1 | P1 sessions differ > 2× in difficulty · P2 gain tracks first-pass WER · P3 no date trend · P4 α robust · P5 calibration worse late | P1–P4 hold, P5 not supported |
| 3a | Q1 gate gains ≥ 0.2 for Qwen and Jev · Q2 agreement leads · Q3 < half of headroom · Q4 softmax ≈ always | Q1, Q2 fail; Q3, Q4 hold |
| 3b | Q5 activations add < 0.1 points | holds (they make it worse) |
| 2 | D3 faithful reproduction · D1 PER rises with days · D2 late PER ≥ 2× released | all hold |
| 4 | E0 pipeline reproduces Cinà · R1 churn ≥ 2× · R2 more points, smaller share · R3 far oracle > 40% · R4 ratio higher far · R5 gate holds near | E0, R1, R3 hold; R2, R4, R5 fail (R4, R5 hold without training-text overlap) |
| T12 | H1 gate halves harm on 2 of 3 arms · H2 headroom ≥ 0.5 | H1 fails (OPT only); H2 holds for OPT and Jev |

| Stamped file | Predictions | Bitcoin block |
|---|---|---|
| `timestamps/PREDICTIONS-a8486e0.md` | Q1–Q5 | 970412 (earliest attestation) |
| `timestamps/PREDICTIONS-1b0cd42.md` | D1–D3 | 970724 |
| `timestamps/PREDICTIONS-41b8dff.md` | E0, R1–R5 | 971147 |
| `timestamps/PREDICTIONS-a204603.md` | H1, H2 | 971469 |

Check a stamp with the OpenTimestamps client: `ots verify timestamps/PREDICTIONS-a8486e0.md.ots` (needs a
Bitcoin node, or compare the proof's Merkle root with a block explorer). The file names carry the commit
hashes of the private working history in which the predictions were made.

## Layout

| Path | What |
|---|---|
| `src/` | one script per step; `src/paths.py` says where the large inputs live |
| `results/` | every result file the paper cites (JSON) |
| `derived/step4/` | the drifted decoder's 100-candidate lists and OPT/Qwen scores, built on a rented GPU (8.2 MB) |
| `paper/` | `paper.md`, `make_figures.py`, `build_pdf.sh`, the figures and the PDF |
| `cloud/` | the RunPod launcher and pod scripts for steps 2 and 4, with their tests; `rest_gpu_names.txt` is RunPod's REST list of GPU type names, which a test checks the launcher's choices against |
| `LAB-NOTES.md` | a dated record of every run, failure and fix |
| `PREDICTIONS.md`, `timestamps/` | predictions and their timestamps |
| `SUMMARY.md`, `ARTICLE-NOTES.md` | working summaries |

## Rerunning the analyses

The analyses read inputs from `$RESCORING_DRIFT_DATA` (default `./data`). Steps 1, 3a and 4 need only
Cinà's repository; the session mapping they use is committed in `results/val_sessions.json` and
`results/t12_sessions.json`.

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
mkdir -p data && git clone https://github.com/gabrycina/how-much-language-model data/cina
git -C data/cina checkout 3001f70
.venv/bin/python -I reproduce.py --rerun
```

`--rerun` re-executes `src/step1_sessions.py`, `src/step1_headroom.py`, `src/step3a_gate.py` (T15 and T12,
with and without `--exclude-train-overlap`) and `src/step4_rescoring.py`, reports which result files changed,
and then runs the checks. About 4 minutes on an M1 Max; on the author's machine all 17 result files came out
byte-identical.

The other steps need the recordings, a GPU or model weights:

| Step | Command | Needs | Cost and time |
|---|---|---|---|
| Session map | `python -I src/sessions.py [--participant t12]` | T15 `data_val.hdf5`; T12 `competitionData/test` | seconds |
| 2, training | `python cloud/runpod.py run --until 2023.11.26 --name early` (or `src/train_rnn.py` locally) | RunPod account; official code at `980a5b7` | 190.8 min on an RTX 4090, $3.72; ~170–250 h on Apple MPS |
| 2, evaluation | `python -I src/step2_drift.py --checkpoint <run>/checkpoint --name early` | T15 `data_val.hdf5`, checkpoint | ~5 min on MPS |
| 5 | `python -I src/step5_review.py` (near-regime stability, random-feature control, feature distributions), `python -I src/step5_input_layer.py` (released checkpoint through the frozen input layer) | Cinà's repository; the second needs the Dryad data and the released checkpoint | ~2 min; ~5 min on MPS |
| review | `python -I src/review_checks.py` (the numbers the review-driven revision cites; needs the T15 training transcripts for the early-model overlap count) | Cinà's repository, Dryad hdf5 | seconds |
| 3b | `HF_HOME=$RESCORING_DRIFT_DATA/hf python -I src/step3b_activations.py`, then `python -I src/step3b_probe.py` | OPT-6.7b at `a45aa65` (~13 GB) | ~11 min, then 2.5 min |
| 4, logits | `python -I src/step4_logits.py` | T15 data, early and released checkpoints | ~8 min on MPS |
| 4, decode and score | `python cloud/runpod.py run --job decode --name step4 --hard-stop 6h` | 3-gram LM (14.1 GB, ~60 GB RAM), RTX 3090 | about 1 h, ~$0.46 (with two smoke runs, $0.83) |

The launcher reads `RUNPOD_API_KEY` (or the macOS keychain item `runpod-api-key`) and deletes every pod it
creates on success or failure; `cloud/test_runpod.py` and `cloud/test_pod_decode.py` exercise those paths
against a simulated API. One idle pod during setup cost $12.57; `LAB-NOTES.md` (2026-10-10) records how.

## Data access

This repository redistributes no recordings. Get them from their owners:

| Data | Source | License | SHA-256 |
|---|---|---|---|
| T15 recordings, `t15_copyTask_neuralData.zip` (11.05 GB) | Dryad [doi:10.5061/dryad.dncjsxm85](https://doi.org/10.5061/dryad.dncjsxm85) | CC0 | `fb828fa7ef937aca405d6912addc876f6747f75a8a1711f6d2052aacf34ec5b5` |
| T15 released checkpoint, `t15_pretrained_rnn_baseline.zip` (485 MB) | same record | CC0 | `04ab2275eeb48abad9b4011dde6f7c9e5dd9f643b8a677c1ba5e06dc53bb4111` |
| 3-gram language model, `languageModel.tar.gz` (14.1 GB) | same record | CC0 | `7b64d36c64765dc0a5e890bb0cc795657c827b7d314fbf5ed5dc8778bd166d7e` |
| T12 recordings, `competitionData.tar.gz` (3.67 GB) | Dryad [doi:10.5061/dryad.x69p8czpq](https://doi.org/10.5061/dryad.x69p8czpq) | CC0 | `5b5c9784d80d5286e69352621366be2e4ff468e043e082e7aa7cbc90d18336b3` |

Dryad's API and scripted downloads require a login, so the official `download_data.py` fails. The Hugging
Face mirror `yw784/brain-to-text-25-competition` carries the T15 files and the language model; its files
matched the digests above, which are the ones Dryad's file listing publishes. Check any copy against them.
The T12 archive was downloaded from Dryad in a browser.

Code that is used but not copied here:

- Cinà's [how-much-language-model](https://github.com/gabrycina/how-much-language-model) at `3001f70`
  (code MIT, text CC BY 4.0): `data/lists_published.pkl`, `data/scores_published/kaldi_{opt,qwen,jev}.pkl`,
  `data/lists_t12.pkl`, `data/scores_t12/t12_{opt,qwen,jev}.pkl`, `data/train_sentences*.txt`, and his
  `analyze.py`, `revision_analysis.py`, `build_lists_kaldi.py` and `runscores_cuda.py`. His `lists_t12.pkl`
  metadata names its source as T15; n = 877 and oracle 8.3% are T12's.
- The official [nejm-brain-to-text](https://github.com/Neuroprosthetics-Lab/nejm-brain-to-text) at
  `980a5b7`. It carries no license file, so none of its code is copied here; the scripts import it from a
  local clone.
- Model snapshots: `facebook/opt-6.7b` `a45aa65bbeb77c1558bc99bedc6779195462dab0` (transformers also fetched
  `3f007b4` on the pod), `Qwen/Qwen2.5-7B-Instruct` `a09a35458c702b33eeacc393d103063234e8bc28`.

The checkpoint trained in step 2 (459 MB) and the step 4 logits (40 MB) are outside this repository;
they are deposited on Zenodo, [doi:10.5281/zenodo.23745520](https://doi.org/10.5281/zenodo.23745520), with a `SHA256SUMS` file.

## Third-party pickles

Every cached pickle from Cinà's repository was listed with `python -m pickletools` and contains no `GLOBAL`
or `STACK_GLOBAL` opcode. `src/step1_sessions.py` loads them, and the files in `derived/step4/`, through
`BuiltinsOnlyUnpickler`, which refuses any importable class. The released RNN checkpoint is loaded with
`torch.load(weights_only=True)` widened by exactly NumPy's scalar and dtype types (`src/step2_drift.py`).
Run anything that reads downloaded files with `python -I`.

## Evaluation rules

Every fitted choice (blend weight, temperature, gate weights and threshold, probe layer and size) is fit on
the earliest sessions only; the random split is reported only next to the temporal one. All candidates of a
trial stay on one side of a split. Results are reported with and without the evaluation sentences that also
occur in the decoder's training text. Word-level claims use explicit alignments. Null results get the same
write-up as positive ones.

## License and citation

Code: MIT (`LICENSE`). Text, figures, results and lab notes: CC BY 4.0 (`LICENSE-TEXT`). Cite the preprint
(`CITATION.cff`) and Cinà's work wherever his lists, scores or findings are used.

Mado Kagerou · [ORCID 0009-0005-5962-9254](https://orcid.org/0009-0005-5962-9254) · mail@kagerou.glass
