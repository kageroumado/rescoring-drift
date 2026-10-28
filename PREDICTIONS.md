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

### Outcome, step 3a (2026-10-08; see LAB-NOTES.md)

Temporal split: Q1 fails (gains 0.00 / 0.18 / 0.00), Q2 fails (agreement leads for OPT only), Q3 holds
(0% / 21% / 0% of headroom), Q4 holds. Not predicted: at equal WER the gate halves harmful corrections.
Q5 cannot be resolved at the 0.1-point bar on T15 alone (intervals ±0.2–0.25).

### Outcome, step 3b (2026-10-08; see LAB-NOTES.md)

Q5 holds: the activation probe adds nothing over the 3a gate for OPT. On the late sessions it is worse by
0.11 points [+0.04, +0.20]; on the random split +0.02 [−0.08, +0.13].

## Step 2: real drift, written 2026-10-10 before the training run

Setting: the official T15 RNN, official code and schedule (`rnn_args.yaml`, 120,000 batches), trained on
one rented CUDA GPU on the 22 sessions dated up to 2023.11.26 only (the "early model"). It is then run
greedily (CTC, no language model) on the validation trials of all 41 sessions. Sessions after 2023.11.26
have no input layer of their own, so they use the last early session's (2023.11.26): a decoder used with no
recalibration. Reference: the released checkpoint, trained on all sessions, each with its own input layer.
Metric: phoneme error rate (PER) per session, word-weighted by phonemes.

- **D3 (gate for the others).** On the 21 early sessions with validation data, the early model's PER is
  within 15% (relative) of the released checkpoint's. If D3 fails, the run is not a faithful reproduction
  and D1–D2 are reported only as a description of that run, not as drift.
- **D1.** On the 20 late sessions, the early model's per-session PER rises with days since 2023.11.26:
  Spearman ρ > 0.5.
- **D2.** The early model's mean PER over the late sessions is at least twice the released checkpoint's on
  the same trials.

The downstream question (does the rescoring stage's churn and the gate's value change when the decoder
has drifted) needs candidate lists from the early model's output; its predictions get written once D3 holds.

### Outcome, step 2 (2026-10-13; see LAB-NOTES.md)

All three hold. **D3:** early sessions 10.05% vs. the released checkpoint's 9.22%, +9.1% relative (bar 15%).
**D1:** ρ = 0.93 (p = 4×10⁻⁹) over the 20 late sessions. **D2:** late-session PER 56.3% vs. 11.2%, a
ratio of 5.0 (bar 2). Not predicted: the drop is immediate and steep. Seven days after the last training
session PER is already 3× the released checkpoint's (14.8% vs. 4.8%); by day 21 it is 36%, by day 103 64%,
and it levels off between 74% and 81% after a year.

## Step 4: rescoring and the gate on a drifted decoder, written 2026-10-13 before any lists exist

Setting: the step 2 early model's logits on every validation trial, decoded with the published pipeline
(Kaldi WFST, OpenWebText 3-gram, the decoding settings of `evaluate_model.py`) to 100-candidate lists, the
way Cinà's lists were made from the released model. Rescorers: OPT-6.7b and Qwen2.5-7B, protocol A. Every
fitted choice (blend weight α, softmax temperature, the step 3a gate and its threshold) is refit on the
early model's lists for the 21 sessions up to 2023.11.26 and tested on the later ones. The gate's form and
features are the step 3a ones, unchanged. Two regimes, fixed now from step 2's PER: **near** = the 5
sessions within 33 days (PER 13–36%), **far** = the 15 sessions from day 91 on (PER 42–81%). Reference:
Cinà's lists from the released model on the same trials.

- **E0 (gate for the others).** Our pipeline on the released model's logits reproduces Cinà's first-pass
  WER on the late sessions within 0.5 points. If E0 fails, R1–R5 describe our pipeline, not his.
- **R1.** Churn rises with drift: the share of sentences where OPT's pick differs from the first pass is
  at least 2× the released model's in the near regime.
- **R2.** In the near regime always-apply lowers WER by more points than it does for the released model
  on the same trials (more errors to fix), and by a smaller share of first-pass WER.
- **R3.** In the far regime the oracle (best of 100) stays above 40% WER: the lists no longer contain the
  sentence, so no rescorer or gate can recover the decoder.
- **R4.** Helped-to-harmed ratio for always-apply (word-aligned, OPT) is higher in the far regime than in
  the near one: a first pass that is mostly wrong has little left to break.
- **R5.** In the near regime the gate, fit on the early sessions, makes fewer harmful corrections than
  always-apply and is within 0.5 points of its WER. A gate that loses more than 0.5 points means its
  features (score margins) shift under drift and it needs recalibrating with the decoder.

### Outcome, step 4 (2026-10-16; see LAB-NOTES.md)

**E0 holds:** our pipeline on the released model's logits gives 7.82% first-pass WER on the 651 late
sentences against Cinà's 7.75% (difference 0.07). **R1 holds:** in the near regime (189 sentences) OPT
changes 34% of first-pass picks on the drifted decoder against 7% on the released one (5.0×). **R2 fails** on
its second clause: always-apply gains more points (OPT 1.64 vs. 0.08, Qwen 1.88 vs. −0.16), but also a larger
share of first-pass WER (OPT 6.9% vs. 2.4%, Qwen 7.9% vs. −4.8%), not a smaller one. **R3 holds:** the far
regime's oracle WER is 76.8% (near 11.3%); the truth is in the list for about half the sentences overall.
**R4 fails:** OPT's word-aligned fixed-to-broken ratio is 1.30 in the far regime against 1.88 in the near
one. **R5 fails** for Qwen and holds for OPT: OPT's gate makes 14 harmful changes vs. always-apply's 15 at
+0.16 WER; Qwen's makes 7 vs. 16 but costs 0.63 points (bar 0.5; session-bootstrap 95% interval +0.09 to
+1.00). Without the 117 late and 11 early sentences that occur in the decoder's training text, all of them
in the far regime or the fit set, E0–R3 are unchanged and **R4 and R5 flip to holding** (far ratio 2.00 vs.
near 1.89; Qwen's gate +0.39 points with 4 harmful changes vs. 16). The near test sentences are identical
in both runs, so R5's flip comes from refitting on 11 fewer early sentences: the verdict is fragile. The
pre-registered verdicts are the full-data ones.

## T12 replication of steps 1 and 3a, written 2026-10-14 before any T12 WER is computed

Data: Cinà's cached T12 lists (`lists_t12.pkl`, 877 sentences, 100-best, acoustic scale 0.3) and rescorer
scores (`scores_t12/t12_{opt,qwen,jev}.pkl`), mapped to session by the `test` split of Dryad's
`competitionData` (SHA-256 verified): 880 trials in 24 sessions, 874 sentences with a unique session. Same
code and every fixed choice of steps 1 and 3a, unchanged; only the inputs differ (`--participant t12`).
Temporal split by the same rule: the 14 sessions 2022.04.28–07.05 (474 sentences) → the 10 sessions
2022.07.14–08.25 (400). Known before writing: the T12 lists are harder than T15's (oracle WER 8.3% vs.
2.8%; the truth is in the list for 71% of sentences vs. 88%).

P1–P4 and Q1–Q4 are re-tested as written above. The decoder saw every T12 session in training, as on T15,
so these again measure session difficulty, not drift. Two predictions carry the replication:

- **H1 (step 3a's unpredicted finding, now predicted).** On the late T12 sessions the logistic gate makes
  at most half as many harmful corrections as always-apply for at least two of the three arms, at a WER
  no more than 0.2 points above always-apply's.
- **H2 (step 1's headroom finding).** The oracle gate's headroom over always-apply on the late sessions is
  at least 0.5 WER points for every arm.

If H1 fails, the harm-halving result is reported as a T15 observation that did not replicate. Pooling T12
with T15 for power is exploratory and is not used to rescue H1.

### Outcome, T12 replication (2026-10-14; see LAB-NOTES.md)

Temporal split, T12 late sessions. **H1 fails:** it holds for OPT (11 harmful corrections vs. always-apply's
30, WER 18.19 vs. 18.40) but not for Jev (the tuned threshold applies all 94 changes, 36 harmful) or Qwen
(α tuned on the early sessions is 0, so it changes nothing and no gate can be fit). **H2 holds for OPT and
Jev** (headroom 1.8 and 2.0 points, larger than T15's) and fails for Qwen for the same α = 0 reason. P1
holds (2.6×). P2 fails (OPT ρ = 0.13). P3 fails (ρ = −0.37, p = 0.07: later T12 sessions are easier). P4
fails for OPT and Qwen (±0.41) and holds for Jev (0.12). P5 holds for all three arms, unlike on T15. Q1
fails, Q2 holds (agreement leads for both fittable arms), Q3 holds (12%, 0%), Q4 fails narrowly (OPT 0.12). Without the
sentences that occur in the decoder's training text (T15: 117 of 651 late sentences, T12: 1 of 400),
every verdict above is unchanged.

## Step 5: analyses asked for in review, written 2026-10-25 before any of them ran

Four controls a reviewer asked for after the manuscript was drafted. They are not part of the registered
program; they are written down before running so that their verdicts are read the same way.

- **S1 (near-regime stability).** With the gate fit once on the drifted decoder's training sessions, the
  paired gate-minus-always-apply difference is computed per near-regime session, and the regime's verdict is
  recomputed leaving each of the five sessions out. Prediction: for OPT, no single session's removal moves the
  gate-minus-always difference across zero; for Qwen, at least one does (its full-regime cost of 0.63 points
  rests on few sentences).
- **S2 (input layer).** The released checkpoint, run on the 20 late sessions through the 2023.11.26 input
  layer instead of each session's own. Prediction: its phoneme-weighted late-session PER lies between the
  released checkpoint's own-layer PER (11.2%) and the early model's (56.3%), and closer to the early model's
  (above 33%): the input layer carries most of the drift damage.
- **S3 (random-feature control).** The step 3b probe is rerun with the activation components replaced by
  Gaussian random features of the same dimensionality, same cross-validated selection, same threshold tuning,
  over 20 seeds. Prediction: the median random-feature gate is also worse than the 3a gate on the late sessions
  (the harm comes from adding dimensions at 90 fit examples), and the real probe's +0.11 points falls inside
  the random distribution's central 90%.
- **S4 (feature distributions).** For the calibrated T15 gate and the near-regime gate, each of the eight
  features' standardized mean for accepted and vetoed changes, and for helpful and harmful ones; plus list
  length on matched sessions (the early model's lists on the 20 late sessions against the released model's
  on the same sessions). Prediction: on the calibrated decoder, rescorer agreement separates helpful from
  harmful changes by at least 0.5 standard deviations; in the near regime it separates them by less than 0.25;
  and the early model's lists on matched late sessions are longer than the released model's.

### Outcome, S1, S3, S4 (2026-10-25, after the run; see LAB-NOTES.md)

**S1 holds for OPT and fails for Qwen.** OPT: gate minus always-apply is +0.16 points over the five near
sessions; per session −0.47, +0.60, 0.00, 0.00, +0.30; leaving one session out gives 0.00 to +0.28, never
below zero. Qwen: +0.63; per session 0.00, +0.90, +0.60, 0.00, +1.20; leaving one out gives +0.42 to +0.76.
Qwen's cost is not carried by any single session.

**S3 holds.** Twenty gates with Gaussian random features in place of the activation components are worse
than the 3a gate by a median of +0.09 points (5th to 95th percentile 0.00 to +0.25; 17 of 20 worse); the real
probe's +0.11 sits inside that range, with 7 of 20 random seeds worse than it.

**S4 holds on two clauses and fails on one.** Rescorer agreement separates helpful from harmful changes by
+0.76 standard deviations on the calibrated decoder for both OPT and Qwen (first clause holds), and by
+0.49 (OPT) and +0.56 (Qwen) in the near regime (second clause, under 0.25, fails: the feature keeps most
of its signal under drift). On the 20 late sessions the early model's lists average 88.5 candidates against
the released model's 39.2 (third clause holds; the all-sessions mean the paper quoted, 59.6, understated it).

### Outcome, S2 (2026-10-25, after the run; see LAB-NOTES.md)

**S2 holds.** The released checkpoint run through the 2023.11.26 input layer has a phoneme-weighted late-session
PER of 45.6%, between its own-layer 11.2% and the early model's 56.3%, and above 33%: the frozen input layer
accounts for about three quarters of the gap between the two checkpoints. The curve has the same shape: 8.8%,
9.1% and 7.8% at days 7, 12 and 14 (own layer 4.8%, 3.5%, 2.9%; early model 14.8%, 14.6%, 13.0%), then 22.9%
at day 21.
