# Predictions

Written and committed before the analysis that tests them runs.

## Step 1: feasibility (exploratory), 2026-10-08

Data: Cinà's cached 100-candidate lists from the published T15 decoder (pretrained RNN, Kaldi WFST,
OpenWebText 3-gram) and his per-candidate rescorer scores, with each sentence mapped to its recording
session through the per-session `data_val.hdf5` files.

The pretrained RNN was trained on the training split of all 45 sessions, with a per-day input layer.
Validation sentences are never later than the training data, so step 1 measures how difficulty varies
across sessions, not drift over time. Drift needs step 2 (retraining on early sessions only).

- **P1.** First-pass WER varies across sessions by more than a factor of 2 between the easiest and
  hardest quartile of sessions (word-weighted, sessions with at least 10 sentences).
- **P2.** Per-session rescoring gain (first-pass WER minus OPT-6.7b protocol-A WER) correlates positively
  with per-session first-pass WER: Spearman ρ > 0.3.
- **P3.** First-pass WER shows no reliable trend with recording date (|Spearman ρ| < 0.3), because the RNN
  saw every session in training.
- **P4.** Tuning the rescoring weight α on the earliest sessions instead of a random dev split changes
  test WER on the latest sessions by less than 0.2 points: the blend weight is robust to which sessions
  tune it.
- **P5.** A sentence-level confidence (softmax over the blended candidate scores, temperature fit on the
  tuning split) is worse calibrated on the latest sessions under a temporal split than on the random
  split: higher expected calibration error.

The step continues to step 2 if P2 holds, or if P5 shows calibration degrading across sessions. If
neither does, the drift framing loses its main motivation and the project narrows.

### Outcome (2026-10-08, after the run; see LAB-NOTES.md)

P1 holds (11.4×). P2 holds as stated but is mechanical: relative gain does not track first-pass WER.
P3 holds. P4 holds (±0.06 points). P5 not supported (small, mixed). The continuation rule is met only
nominally through P2; the project continues because of a result the predictions didn't anticipate: the
gate headroom (0.5–1.3 points) exceeds the rescoring stage's net gain.

## Step 3a: a correct-or-keep gate from output features, 2026-10-08

Setting: for each sentence where a rescorer's pick differs from the first pass, a gate decides whether
to apply the change or keep the first pass. Primary evaluation is temporal: fit on the 21 earliest
sessions, test on the 20 latest (the step 1 split). The random 30/70 split is reported alongside.

Gate, fixed now: an L2-regularized logistic regression (C = 1, features standardized on the training
split) trained on changed sentences labeled helped vs. not helped, with a probability threshold tuned
on the training split by WER. Features: the first-pass score margin between the first-pass top and the
rescorer's pick; the rescorer score margin; the pick's rank in the first-pass list; the word edit
distance between the two hypotheses; sentence length in words; list length; how many of the three
rescorers (OPT-6.7b, Qwen2.5-7B, Jev) chose the same hypothesis; Cinà's softmax confidence for the arm.

Baselines: no rescoring, always apply (the published pipeline), a threshold on Cinà's softmax
confidence (tuned on the training split), and the oracle (upper bound).

- **Q1.** On the late sessions, the gate lowers WER relative to always-apply by at least 0.2 points for
  Qwen2.5-7B and Jev, and by less than 0.2 for OPT-6.7b (fewer changes to gate).
- **Q2.** Agreement among the three rescorers has the largest standardized coefficient for at least two
  of the three arms.
- **Q3.** The gate recovers less than half of the oracle's headroom over always-apply, for every arm.
- **Q4.** The softmax-confidence threshold is within 0.1 points of always-apply for every arm.

## Step 3b: rescorer activations, written ahead of the run

- **Q5.** A linear probe on OPT-6.7b residual-stream activations at each candidate's final token, with
  the same training split and labels, adds less than 0.1 points over the step 3a gate for OPT on the
  late sessions. The bar a positive result has to clear is set here.
