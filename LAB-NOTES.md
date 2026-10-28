# Lab notes

## 2026-10-08 · Setup

- Created this repo. Cloned Cinà's repository (commit `3001f70`) and the official nejm-brain-to-text repo to
  `/Volumes/Ugreen/Projects/rescoring-drift/`.
- **Prior work re-read.** Cinà's repository already has an output-based abstention analysis
  (`paper/tab_abstention.tex`, `src/revision_analysis.py`): coverage against sentence error and calibration
  error per rescorer. Its dev/test split is a random 30/70 over sentences, with no session grouping. Our
  question narrows to drift across sessions and to the rescorer's internal representations.
- **Pickle audit.** None of the 30 cached pickles (`lists_*.pkl`, `scores_*/*.pkl`) contains a
  GLOBAL or STACK_GLOBAL opcode; they hold only builtin types, so they are safe to load.
- **The cached lists are keyed by sentence text, with no session.** `lists_published.pkl` is a tuple whose
  dict maps the reference sentence to its candidates. Session has to be recovered from the per-session
  `data_val.hdf5` files, which is how Cinà's `src/extract_logits.py` built the lists.
- **Dryad downloads now need a login.** The API returns "Unauthorized, must have current bearer token",
  and the web file stream returns 403 to scripts, so `download_data.py` in the official repo no longer
  works. Hugging Face mirror `yw784/brain-to-text-25-competition` hosts the same files, and its LFS SHA-256
  for each file matches the digest Dryad's file listing publishes. Downloading
  `t15_copyTask_neuralData.zip` (11.05 GB) and `t15_pretrained_rnn_baseline.zip` (485 MB) from it.

## 2026-10-08 · Reproducing Cinà's T15 numbers

- Project environment: `uv venv --python 3.12 /Volumes/Ugreen/Build/rescoring-drift/.venv`, then
  `uv pip install -r requirements.txt` (torch 2.14.1, MPS available).
- **The README's reproduce commands point at paths the repo doesn't ship.** `src/make_results.py` and
  `src/revision_analysis.py` read `real/lists_real.pkl` and `real/scores/`, and `make_results_real.py` reads
  `~/b2t/work/`; the published data sits under `data/`. `src/prereg_analysis.py` takes paths as arguments
  and works:

  ```bash
  cd /Volumes/Ugreen/Projects/rescoring-drift/cina/src
  /Volumes/Ugreen/Build/rescoring-drift/.venv/bin/python prereg_analysis.py --lists ../data/lists_published.pkl \
    --scores ../data/scores_published --prefix kaldi_ --train-text ../data/train_sentences.txt \
    --out /Volumes/Ugreen/Projects/rescoring-drift/cina_prereg_t15.json
  ```
  2 min 38 s on CPU (M1 Max).
- **Every number matches the paper.** Random split 30/70 (seed 0): 419 dev, 978 test. First pass
  8.09% test WER (λ=0.3). OPT-6.7b 7.80% (protocol A) / 7.20% (B, re-tuned λ and α); Qwen2.5-7B 7.75% /
  7.42%; Jev 7.46% / 6.93%. The choice-format controls (optchoice, qwenchoice, Laya) select α=0 or do worse
  than no rescoring. 107 test sentences appear in the RNN's training text; without them the Jev
  equivalence is only partly confirmed.

## 2026-10-08 · Step 1: sessions, predictions P1–P5, gate headroom

- Download finished; both zips match Dryad's SHA-256. Extracted only the 41 `data_val.hdf5` files
  (`unzip -o t15_copyTask_neuralData.zip 'hdf5_data_final/*/data_val.hdf5'`).
- `python -I src/sessions.py` → `results/val_sessions.json`: 1,426 validation trials across 41 sessions.
  1,397 distinct sentences; **17 sentences were spoken in more than one session (45 trials)**, which is why
  Cinà's 1,397 lists cover 1,426 trials: duplicates overwrote each other in his text-keyed cache. 1,380
  lists map to exactly one session; the 17 ambiguous ones are excluded from per-session analysis.
- `python -I src/step1_sessions.py` (8 s) → `results/step1_sessions.json`. Against `PREDICTIONS.md`:
  - **P1 holds, strongly.** Hardest-quartile sessions average 17.6% first-pass WER and the easiest 1.5%
    (11.4×); three sessions have zero errors.
  - **P2 holds as stated but is mechanical.** Gain correlates with first-pass WER (ρ 0.39 OPT, 0.41 Qwen,
    0.56 Jev), yet the *relative* gain (gain ÷ first-pass WER) does not (ρ −0.00, 0.20, −0.12; all p > 0.2).
    The rescorer removes about the same share of errors in easy and hard sessions; hard sessions have
    more to remove. Natural session-to-session variation does not reproduce Cinà's "the stage matters
    more for a weak decoder" beyond that proportionality.
  - **P3 holds.** No date trend in first-pass WER (ρ 0.11, p 0.52), as expected for a decoder trained on
    every session.
  - **P4 holds.** Tuning α on the earliest 21 sessions instead of a random dev split changes late-session WER
    by −0.06 (OPT), 0.00 (Qwen), +0.06 (Jev) points.
  - **P5 not supported.** Calibration error on the late sessions under the temporal split vs. the random
    split: OPT 0.170 vs 0.160, Qwen 0.127 vs 0.117, Jev 0.073 vs 0.089. Small, mixed in sign, no intervals,
    and computed on different evaluation sentences. The temperature fit hits the grid's lower edge
    (T = 0.25) for OPT and Qwen, so their calibration is limited by Cinà's grid.
- **Gate headroom** (`python -I src/step1_headroom.py` → `results/step1_headroom.json`; 966 test sentences,
  6,449 words, first pass 8.00%). The oracle keeps a rescorer's change only where it helps, so it is an
  upper bound, never a result:

  | Rescorer | WER | Oracle-gate WER | Changes helped / harmed / neutral | Words fixed / broken |
  |---|---|---|---|---|
  | OPT-6.7b | 7.71 | 7.26 | 41 / 26 / 22 | 48 / 29 |
  | Qwen2.5-7B | 7.66 | 6.68 | 70 / 55 / 43 | 85 / 63 |
  | Jev | 7.38 | 6.12 | 100 / 73 / 49 | 121 / 81 |

- **Decision.** The drift framing loses its step 1 motivation: with this decoder, sessions differ in
  difficulty but not in how the rescorer behaves. Real drift (a decoder used on sessions after its training)
  is untested and needs step 2. The stronger thread is the churn: each rescorer breaks almost as many
  words as it fixes, and a perfect gate would gain 0.5–1.3 points, more than the stage's whole net
  contribution. Step 3 (can a gate find the harmful changes, from outputs or activations) is now the main
  question; step 2 becomes the test of whether gates hold up under real drift.
- **Deviation from the written P5.** `PREDICTIONS.md` specifies a softmax over the *blended* candidate
  scores; `step1_sessions.py` used Cinà's `softmax_conf` over the rescorer's own scores, for comparability
  with his abstention table. P5's outcome therefore tests his confidence, not the one predicted. Rerun with
  the blended confidence before citing P5 either way.

## 2026-10-08 · Step 3a: correct-or-keep gate from output features

- Predictions Q1–Q5 committed in `a8486e0` and stamped with OpenTimestamps
  (`timestamps/PREDICTIONS-a8486e0.md`, SHA-256 `6a72b713…13af`) before the run.
  Proof upgraded later the same day and anchored in Bitcoin blocks 970412 (00:29:42Z) and 970415
  (01:04:35Z); both merkle roots checked against blockstream.info, since no local node runs `ots verify`.
  The commit is dated 00:10:52Z.
- **Bug found and fixed before reading results.** Cinà's `revision_analysis.fit_temperature` calls
  `analyze.picks` with the module default `ACOUSTIC_SCALE = 0.35`, while the published decoder setting (λ_A
  in his own `prereg_analysis.py`) and all our computations use 0.3. `step1_sessions.fit_temperature` is a
  local copy at 0.3. Rerunning step 1 and 3a changed no reported number: the temperatures fall on the same
  grid points.
- `python -I src/step3a_gate.py` (9 s) → `results/step3a_gate.json`. Late-session test, temporal split
  (primary), word error %; bracket = 95% session-cluster bootstrap of gate minus always-apply:

  | Arm | None | Always apply | Softmax threshold | Gate | Oracle | Gate − always | Changes applied (helped / harmed) |
  |---|---|---|---|---|---|---|---|
  | OPT-6.7b | 7.75 | 7.08 | 7.15 | 7.08 | 6.63 | [−0.23, +0.22] | always 43/18 → gate 33/8 |
  | Qwen2.5-7B | 7.75 | 7.08 | 7.08 | 6.90 | 6.21 | [−0.42, +0.07] | always 57/34 → gate 49/17 |
  | Jev | 7.75 | 6.90 | 6.90 | 6.90 | 6.63 | [0, 0] (threshold 0: applies all) | 43/10 → 43/10 |

  Random split (secondary): gate − always = −0.73 [−1.20, −0.33] OPT, −0.29 [−0.61, +0.05] Qwen, −0.36
  [−0.60, −0.11] Jev.
- **Against the predictions (temporal split):**
  - **Q1 fails.** Gains of 0.00 (OPT), 0.18 (Qwen, interval includes 0) and 0.00 (Jev); none reaches 0.2.
  - **Q2 fails.** Rescorer agreement is the top coefficient only for OPT; for Qwen sentence length leads,
    for Jev edit distance.
  - **Q3 holds.** Headroom recovered: 0%, 21%, 0%.
  - **Q4 holds.** The softmax threshold is within 0.1 of always-apply for every arm, on both splits.
- **What the gate does do: it halves harmful corrections at equal WER.** OPT on late sessions: harms 18 → 8
  while helps 43 → 33; Qwen: harms 34 → 17, helps 57 → 49. Word error is unchanged or slightly lower, but
  the errors left are the decoder's own rather than words the language model put in.
- **The random-split OPT result is an α artifact.** Removing the 17 session-ambiguous sentences from
  Cinà's 419-sentence dev split moves OPT's tuned α from 0.3 to 0.5. At 0.5, OPT changes 202 test sentences
  and nets almost nothing (8.00 → 7.89), and the gate's 55% headroom recovery largely undoes the bad α.
  Tuning the blend weight on ~400 sentences is fragile.
- **Power is the binding constraint.** The late sessions hold 63–119 changed sentences per arm, so
  intervals on gate − always are ±0.2–0.25 points. Q5's 0.1-point bar cannot be resolved on T15 alone.
  Cinà's repository also has T12 lists and scores (`data/lists_t12.pkl`, `scores_t12/`), a second
  participant; pooling them or replicating there is the way to more changed sentences.

## 2026-10-08 · Step 3b: OPT-6.7b activation probe

- `HF_HOME=/Volumes/Ugreen/Projects/rescoring-drift/hf python -I src/step3b_activations.py`: OPT-6.7b at
  commit `a45aa65` (PyTorch `.bin` only, loaded with transformers' `weights_only` loader), float16 on MPS.
  601 distinct hypotheses (first-pass and OPT picks for every sentence OPT changes under either split's α).
  Extraction 67 s; model loading dominated the ~11 min wall time. Output
  `/Volumes/Ugreen/Projects/rescoring-drift/activations/opt_step3b.npz` (73 MB), metadata in
  `results/step3b_activations_meta.json`.
- **The setup reproduces Cinà's OPT scores.** Rescoring 30 full candidate lists: median Spearman 0.9998
  (minimum 0.994) against his cached scores, despite float16 on MPS vs. his bfloat16 on CUDA.
- **Operator error, no effect on results.** I misread a log line (the download job's "exited") as the
  extraction crashing, and started a second copy that ran concurrently on the GPU for ~10 min before I
  noticed and killed it. The completed output is from the first, uninterrupted run.
- `python -I src/step3b_probe.py` (2.5 min) → `results/step3b_probe.json`. Probe = PCA of
  h(OPT pick) − h(first pass) + the 3a features, logistic, layer and components by 5-fold session-grouped CV
  on training only. OPT, word error %:

  | Split | Always | 3a gate | Probe, final token | Probe − 3a gate [95% session bootstrap] | AUC helped: 3a / probe |
  |---|---|---|---|---|---|
  | Temporal (primary) | 7.08 | 7.08 | 7.19 (layer 20, 8 PCs) | **+0.11 [+0.04, +0.20]** | 0.63 / 0.64 |
  | Random | 7.89 | 7.16 | 7.18 (layer 32, 2 PCs) | +0.02 [−0.08, +0.13] | 0.79 / 0.81 |

- **Q5 holds, and more strongly than predicted.** The activations add nothing; on the late sessions they
  make the gate worse, with an interval that excludes zero. Discrimination barely moves (AUC 0.63 → 0.64);
  the threshold tuned on early sessions transfers worse once 8 activation components are in the model,
  the usual cost of adding dimensions with ~90 training examples.
- Exploratory (mean pooling, not predicted): on the late sessions it applies only 30 changes, 22 helpful
  and 1 harmful, but misses enough helpful ones that WER is 7.21. A very conservative gate, which matters
  if a harmful correction costs more than a missed one. Not a result until the cost ratio is fixed in advance.
- **Where step 3 leaves the project.** The rescorer's internals hold no usable signal about which of its
  own corrections are wrong, at this sample size, beyond what its outputs already say. The output-feature
  gate does not lower late-session WER reliably, but it halves harmful corrections at equal WER. Remaining
  questions: (1) replicate on T12 (Cinà's second participant) for power; (2) state the gate's value as a
  harm/miss trade-off with costs fixed in advance; (3) step 2, real drift, is still untested.

## 2026-10-08 · Toward step 2 (real drift) and the T12 replication

- **Heavy runs now go through idle jobs** (a local scheduler): a job starts only after the Mac has had no
  input for its threshold, freezes on keyboard or mouse input, resumes after the threshold, and reports
  with the log when it ends.
- **Training on Apple GPU works.** `src/train_rnn.py` wraps the official `rnn_trainer.py` without touching
  its training logic. Needed: `torchaudio` (edit distance), `torch.compile` and bfloat16 autocast off (CUDA
  paths), and gradient clipping without the foreach kernel ("can't use the foreach API on mps tensors";
  same result). Smoke test, 3 batches on the 22 sessions up to 2023.11.26: runs, validates and
  checkpoints in 1.1 min, nearly all of it validation.
- Training data extracted: `hdf5_data_final/*/data_train.hdf5` (45 sessions, 11 GB with the val files).
  The early model covers 22 sessions, because 2023.08.11 has training data but no validation file.
- **Queued:** idle job `5acb0a`, a 300-batch benchmark, to measure seconds per batch before committing
  to the official 120,000-batch schedule. Step 2 predictions get written and timestamped after the benchmark
  and before the full run.
- **T12 needs Dryad's `competitionData.tar.gz`** (3.67 GB, SHA-256 `5b5c9784…36b3`) for its session labels:
  Cinà's T12 lists (877 sentences) are keyed by sentence text, like T15's. No checksum-verifiable mirror
  was found (`eminorhan/willett` on Hugging Face is a reprocessed Parquet copy), and Dryad downloads need
  a login, so it is downloaded by hand in a browser.

## 2026-10-08 · RNN speed benchmark: the official schedule does not fit on this Mac

- **Ran** (idle job `5acb0a`, 39 min wall, no pauses): `python -I src/train_rnn.py --name bench300 --batches 300
  --val-every 100000 --sessions-until 2023.11.26`. Log: the idle job's log (`job-5acb0a.log`); checkpoint in
  `/Volumes/Ugreen/Projects/rescoring-drift/runs/bench300/`.
- **Speed on MPS: ~7.5 s per batch of 64.** The trainer's own step timer reads 7.3 s at batch 200 (data loading
  excluded); wall clock is 7.7 s/batch over batches 0–200 and 7.4 s/batch over 200–299. Validation over the 21
  early val sessions takes 27–37 s.
- **The full 120,000-batch schedule would need ~250 h (~10 days) of uninterrupted idle compute.** The official
  run is a few hours on one CUDA GPU, so MPS is roughly two orders of magnitude slower here. No MPS-fallback
  warning appears in the log, so CTC loss is not being bounced to CPU. The likely cost is the 5-layer, 768-unit
  GRU, which MPS executes step by step.
- The log's `Using device: cpu` and `Using torch.compile` lines come from the trainer's constructor before the
  wrapper patches it. The wrapper's own line confirms `300 batches on mps`, and `torch.compile` is replaced by
  the identity.
- Learning sanity check: loss 739 → 79, val PER 1.22 → 0.99 (still in the blank-output phase, as expected at 300 batches).
- **Next, queued:** idle job `dc6e6e`, 41 batches on CPU and then 41 on MPS (`--device`, `--log-every 10`, both
  new wrapper flags), to see whether the 10-core CPU beats MPS on this GRU. Then pick one of: a shorter schedule
  with the learning-rate decay compressed to match (it has to be stated as a deviation, and its PER on early
  sessions has to land near the official model's before drift is measured on it); the MacBook as a second host;
  or rented CUDA time (~$5 for one run). Step 2 predictions wait until the schedule is fixed.

## 2026-10-08 · CPU vs MPS: MPS wins by ~6×, so the schedule question stands

- **Ran** (idle job `dc6e6e`, 38 min, no pauses): `train_rnn.py --batches 41 --log-every 10 --val-every 100000
  --sessions-until 2023.11.26`, first with `--device cpu` and then with `--device mps`. Log:
  the idle job's log (`job-dc6e6e.log`).
- **CPU (10 cores): 26–43 s per training step**, ~30 s/batch over batches 10–40 of wall clock, and 5.4–7.0 min per
  validation pass. **MPS: 4.3–5.6 s per step**, 5.3 s/batch wall clock over batches 0–40, and 25–34 s per
  validation pass. The MPS steps here are faster than the 7.3 s in `bench300`. Batch cost varies with sentence
  length, so ~5–7.5 s/batch is the honest range.
- Both devices produce the same numbers to the second decimal (loss 545–547 at batch 40, val PER 0.873/0.876).
  The device changes speed only. (Val PER is 0.87 at batch 40 here but 0.99 at batch 300 in `bench300`; early
  CTC training passes through an all-blank phase, so neither number says anything about the final model.)
- **120,000 batches on MPS ≈ 170–250 h.** The three options in the entry above remain open. Rented CUDA
  (~$5 for a few hours) is the only one that keeps the official schedule without tying up the Studio for over a
  week. It spends money, so it waits for a decision.

## 2026-10-10 · Step 2 set up for rented CUDA

- A RunPod account was created ($15 credit). The official 120,000-batch schedule (7–10 days on the Studio's
  GPU) goes to one RTX 4090 on Community Cloud.
- **Predictions D1–D3 committed before the run** (`1b0cd42`) and stamped with OpenTimestamps.
- `cloud/pod_train.sh` (runs on the pod): downloads `t15_copyTask_neuralData.zip` from the Hugging Face mirror
  and checks it against Dryad's SHA-256, extracts the sessions up to 2023.11.26, clones nejm-brain-to-text at
  `980a5b7`, and runs the official `train_model.py` unmodified. Only `rnn_args.yaml` changes: session list,
  paths, `gpu_number: '0'`. Hard limit 12 h. It writes `DONE` with the exit status, and stops the pod itself
  if nothing fetches the run within 2 h.
- `cloud/runpod.py run` (runs here, in the background, not as an idle job, since pausing it would only delay
  deleting a billing pod): REST API v1, image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`,
  a dedicated SSH key `~/.ssh/id_ed25519_runpod`, poll every 5 min, `scp` the run to
  `/Volumes/Ugreen/Projects/rescoring-drift/runs/early-cuda/`, delete the pod on success or any failure,
  then wake this session through `remind`. Untested until the API key exists.
- `src/step2_drift.py`: greedy CTC per session; sessions after training use the last trained session's
  input layer. Standard CTC collapse (merge repeats, then drop blanks); the official `evaluate_model.py` drops
  blanks first. The released checkpoint's pickle imports only tensors, an OrderedDict and NumPy
  scalar/dtype types (listed with `pickletools`), so it loads with `weights_only=True` widened by exactly
  those. Reference evaluation of the released checkpoint queued as idle job `edd82e`.
- **Blocked on** the RunPod API key (to go in the keychain as `runpod-api-key`).

## 2026-10-10 · Step 2 reference: the released checkpoint's PER per session

- **Ran** (idle job `edd82e`, 5 min on MPS, no pauses): `python -I src/step2_drift.py --checkpoint
  /Volumes/Ugreen/Projects/rescoring-drift/dryad/t15_pretrained_rnn_baseline/checkpoint --name released`
  → `results/step2_per_released.json`.
- **Greedy PER, phoneme-weighted: 10.19% over all 41 sessions; 9.22% on the 21 sessions up to 2023.11.26, 11.17%
  on the 20 after.** Per session 1.4–30.8%, median 9.2%. Every session here was in the model's training, so
  the early/late gap is session difficulty, not drift (the same finding as step 1's P3).
- **The evaluation reproduces the official one.** The released training log reports best average val PER
  0.1010 (ours 10.19% phoneme-weighted), and 0.2149 / 0.2111 on 2025.03.30 / 2025.04.13 (ours 21.6% / 21.4%).
  The residual comes from the collapse order (standard CTC here, blanks-first in `evaluate_model.py`), float32
  vs. bfloat16, and session averaging vs. phoneme weighting.
- **The official schedule took 213 min on one RTX 4090** (training log: `cuda:1`, 17:24 → 20:57 on
  2025-06-30). So the rented run is ~4 h with the data download, about $1.50–3 on Community Cloud.

## 2026-10-10 · Launching the rented run: three problems before training

- **The key works, but Cloudflare blocks Python's default user agent.** `GET /v1/pods` from `urllib` answers
  HTTP 403 and the same request from curl answers 200. The launcher now sends `User-Agent: rescoring-drift/1.0`.
- **No RTX 4090 on Community Cloud** (`create pod: There are no instances currently available`). The launcher
  now takes the first available of 4090, A5000, 3090, L40S, A6000, A40 and RTX 6000 Ada (`gpuTypePriority:
  custom`), Community before Secure, retrying every 5 min for up to 2 h.
- **The first host had a broken GPU.** Pod `r6ifuzlp6wnagc` (L40S, $0.79/h) listed its GPU in `nvidia-smi`,
  but PyTorch reported `CUDA unknown error` and `cuda False`. The official trainer falls back to CPU on that,
  and would have spent the 12 h limit (~$9.50) producing nothing. Caught 3 min in from the pod's log;
  monitor killed and pod deleted (~$0.05). Two guards now: the launcher asks PyTorch for CUDA after boot and
  replaces the host, up to 3 times; `pod_train.sh` refuses to train without CUDA (exit 90).
- **Community hosts fail CUDA init; Secure Cloud works.** Three Community pods (L40S and 4090s, driver
  580.126.09, two machines) all reported `CUDA unknown error`. It isn't the image: the CUDA 12.9 image
  `runpod/pytorch:1.4.0-cu1290-torch280-ubuntu2204` fails identically there. A Secure Cloud 4090 (driver
  595.91.07) answers `cuInit 0` and PyTorch sees it. The launcher uses Secure only, at $0.89/h.
- **Starting the pod job over SSH hung.** `nohup … &` with stdin still attached keeps the SSH session open
  until the job ends; the 120 s timeout fired and the pod was deleted. The start is now
  `setsid nohup … </dev/null >/dev/null 2>&1 &`, followed by a `pgrep` check that the job is alive. One Secure pod
  also never became reachable within 20 min and was replaced. Diagnostics and failed launches so far: ~$0.85.
- **Running since 09:31** on Secure pod `mjrf4pmnripfhc` (RTX 4090, torch 2.8.0+cu129, CUDA true), $0.89/h. On the
  way: a Secure host with the same broken CUDA (caught by the launcher check, $0.07) and the start command
  hanging again. A diagnostic pod showed a plain `setsid nohup sleep` returning in 1.2 s, while
  `pod_train.sh` still holds the SSH channel (likely its `exec > >(tee …)` logging). So the launcher now treats a slow
  start as non-fatal and checks the job with `pgrep`. All `ssh`/`scp` calls get `-n` and a `/dev/null` stdin.

## 2026-10-10 · The rented run idled for 14 h: $12.57 lost, and the guards that were missing

- **What happened.** Pod `6mfvoddjontex4` (Secure 4090, $0.89/h) booted at 09:42. In `pod_train.sh`, `git clone` of
  nejm-brain-to-text failed mid-transfer (`fatal: expected flush after ref listing`). Under `set -e` the script
  exited without writing `DONE`, so its own 2 h fetch-grace stop never ran either. The monitor logged that same
  fatal line every 5 min, 168 times, and kept waiting until its 14 h limit, then deleted the pod at 23:45.
  Cost $12.57. With the earlier diagnostics, ~$13.60 of the $15 credit is spent and **$1.44 remains**
  (GraphQL `myself.clientBalance`).
- **Why the guards failed.** Each one assumed the script reaches its end: the 12 h `timeout` wrapped only
  training, the self-stop came after training, and the monitor waited for `DONE`. Nothing covered a script
  that died early, and the monitor did not read the error it was printing.
- **Fixes (committed with this entry):**
  1. The pod schedules its own `runpodctl stop pod` 15 h after boot, before anything else runs.
  2. An `EXIT` trap writes `DONE` with the status on any exit, including failures under `set -e`; tested
     locally with a forced failure.
  3. The monitor fails fast on a failure line (`fatal:`, `Traceback`, `Error:`, `Killed`) or a log tail
     unchanged for 30 min, then fetches the log and deletes the pod.
  4. No network clone on the pod: the launcher uploads `nejm.tar` (the local clone at `980a5b7`), and the
     script verifies the commit.
- **Not rerun:** $1.44 cannot cover a ~$4 run. A top-up is a spending decision, so the run waits.

## 2026-10-12 · Testing the cloud pipeline before the full run

The account was topped up after the 14 h loss, on the condition that everything be tested before deploying. Four stages, cheapest
first; each stage had to pass before the next. Real-pod tests cost ~$0.85 in total (balance $11.43 → $10.60).

1. **Simulated RunPod, local, free** (`cloud/test_runpod.py`, 10 tests): normal completion, a training crash,
   the 14 h incident (fatal line, no DONE), a silent stall, no CUDA on any host, never launched, delete retried,
   delete never succeeding, a start command that hangs while the job runs, the pod environment. Each must
   delete every pod it created, wake its owner exactly once, and keep pod log text out of the wake.
   **Mutation check:** with the failure-line and stall guards disabled, the incident scenarios fail at exactly
   50,400 s, the real 14 h, so the tests do catch it.
2. **Self-stop on a real pod** (`runpod.py selfstop-test`): **the guard had never worked.** Processes started
   over ssh lack `RUNPOD_POD_ID` and `RUNPOD_API_KEY`, `runpodctl` was unconfigured, and the `kill 1` fallback did
   nothing. Fix: `source /etc/rp_environment` (RunPod's own copy of the container environment), a REST `stop`
   fallback, and the script refuses to start without both values. Rerun: the pod stopped itself at
   HARD_STOP=3m (EXITED), no help ($0.04).
3. **Deliberate failure on a real pod** (`--until 1999.01.01`): the failure path worked (exit 1, train.log
   saved, pod deleted, woken; $0.15) and **caught a bug the full run would have hit**: git refused the
   uploaded repo (`dubious ownership`; the tar carried the Mac's uid). A slow host then timed out uploading
   the 224 MB repo archive ($0.27). The archive is now `model_training/` plus a `COMMIT` file verified on the Mac
   (150 KB), with an import check of the trainer before the 11 GB download.
4. **End-to-end smoke on a real pod** (`--smoke-batches 60`): official trainer on CUDA (`cuda:0`,
   `torch.compile`), 60 batches, two validations, checkpoint saved, exit 0, run fetched, pod deleted ($0.13).
   **The local load caught one more bug:** NumPy 2 on the pod pickles `numpy._core.multiarray.scalar`, while
   the weights-only allowlist had the NumPy 1 spelling. Both are allowed now; the smoke and released
   checkpoints both load and decode a late-session trial. The healthy smoke log contains none of the failure
   markers, so the monitor won't kill a good run.

Timing: the official log runs ~0.1 s/batch after a ~50 s warm-up for the first 200 batches (213 min total); the
smoke run's 60 batches in ~26 s match that warm-up. Expected full run: ~3.5 h on a 4090, under 6 h on the
slowest listed GPU; the limits are 12 h of training and a 15 h pod self-stop.

## 2026-10-12 · Step 2: the early model trained

- **Ran** on RunPod Secure Cloud, pod `j79j1rgn1xwsj4` ($1.09/h), official `rnn_args.yaml` with the
  sessions cut at 2023.11.26: 120,000 batches in **190.8 min**, training exit 0. Launcher state:
  `/Volumes/Ugreen/Projects/rescoring-drift/runs/cloud-state.json`; checkpoint and train log in
  `/Volumes/Ugreen/Projects/rescoring-drift/runs/early-cuda/`. Pod deleted by the launcher, confirmed through
  the API (0 pods). **Cost $3.72.** The monitor guards stayed quiet: no failure markers, the log never stalled.
- **The train log already points at D3 passing:** best average val PER 0.0991 over its 21 sessions (session
  average, blanks-first collapse). The released checkpoint scored 9.22% phoneme-weighted on the same sessions
  with our evaluation. The two numbers use different averaging, so D3 is decided only by the same script
  on both: idle job `587891` (`src/step2_drift.py --name early`, ~5 min on MPS).
- Loss reached ~0.01 by batch 40,000 and stayed there; val PER on 2023.11.26 went 9.4% → 7.6% → 7.5%
  over the run, so the later batches still improved held-out trials despite the near-zero train loss.

## 2026-10-13 · Step 2: the early model drifts, D1–D3 all hold

- **Ran** (idle job `587891`, 5 min on MPS, no pauses): `python -I src/step2_drift.py --checkpoint
  /Volumes/Ugreen/Projects/rescoring-drift/runs/early-cuda/early/checkpoint --name early` →
  `results/step2_per_early.json`. Compared against `results/step2_per_released.json`, phoneme-weighted.
- **D3 holds:** 10.05% on the 21 early sessions vs. 9.22% released (+9.1% relative, bar 15%). The run is a
  faithful reproduction, so the late sessions measure drift.
- **D1 holds:** Spearman ρ = 0.93 between late-session PER and days since 2023.11.26 (p = 4×10⁻⁹).
- **D2 holds:** 56.3% vs. 11.2% on the late sessions, 5.0×.
- Per session (days after training: early model / released): 7 d 14.8 / 4.8 · 12 d 14.6 / 3.5 · 14 d
  13.0 / 2.9 · 21 d 36.4 / 8.6 · 33 d 36.3 / 6.2 · 91 d 41.7 / 6.6 · 103 d 63.9 / 18.2 · 166 d 63.3 / 10.0 ·
  245 d 71.0 / 9.0 · 411 d 77.8 / 24.2 · 504 d 73.8 / 21.4.
- **What this means for the downstream question:** past ~3 months the decoder's output is mostly wrong
  (PER > 60%), so there is little left for a rescorer to choose between. The informative range for
  rescoring churn and the gate is the first month (13–36% PER), where five sessions sit (12.03–12.29).
  The downstream predictions should be written per regime, not as one late-session mean.
- **Next:** write and timestamp the downstream predictions, then n-gram decode the early model's logits to
  100-candidate lists (needs the 3-gram LM, ~60 GB RAM, so a pod; same test regimen before the full run).
- **Step 4 predictions committed and stamped** (41b8dff, `.ots` pending, upgrade reminder `83d018`):
  E0 (our decode reproduces Cinà's first pass within 0.5 WER points) gates R1–R5 (churn ×2 near, larger
  absolute but smaller relative gain near, oracle > 40% far, helped:harmed higher far than near, gate
  holds near). `ots` lives in the project venv: `/Volumes/Ugreen/Build/rescoring-drift/.venv/bin/ots`.
- **Before the pod:** write the decode script and run it locally on a handful of trials with a small LM
  (the 1-gram/5-gram-free path) so the pod run only swaps in the 3-gram; asked for approval before spending.

## 2026-10-14 · T12 replication of steps 1 and 3a

- **Data:** Dryad's `competitionData.tar.gz` downloaded in a browser; SHA-256 `5b5c9784d80d…d18336b3` matches the
  digest Dryad's API publishes for it. Moved to `/Volumes/Ugreen/Projects/rescoring-drift/t12/` and
  extracted (train 5.3 GB, test 517 MB, competitionHoldOut 700 MB). Cinà's 877 T12 lists match exactly the
  880 trials of the `test` split (24 sessions); the holdout split has no transcripts and train matches 25.
  His `lists_t12.pkl` metadata says "T15" in `source`, a copy-paste slip: n = 877 and oracle 8.3% are T12's.
- **Code:** `src/sessions.py --participant t12` → `results/t12_sessions.json` (874 sentences with a unique
  session); `step1_sessions.py` and `step3a_gate.py` take `--participant t12` and write `_t12` results. Re-ran
  both on T15 first: outputs byte-identical to the committed ones. Predictions H1–H2 committed (a204603) and
  stamped (77ee621) before the T12 run.
- **Ran:** `python -I src/step1_sessions.py --participant t12`, `python -I src/step3a_gate.py --participant t12`
  (seconds each, CPU) → `results/step1_sessions_t12.json`, `results/step3a_gate_t12.json`.
- **Broke once:** step 3a crashed on Qwen, because α tuned on the early T12 sessions is 0.0, so Qwen changes
  no sentence and the logistic fit gets an empty matrix. The script now records such an arm as "gate not
  fittable"; T15 output unchanged.
- **Results, temporal split (14 sessions → 10, 474 → 400 sentences):**
  - Rescoring nets nothing on the harder participant. First pass 18.40% WER; always-apply OPT 18.40
    (33 helped, 30 harmed), Jev 18.61 (34 helped, 36 harmed). Step 1's central finding replicates.
  - The gate's harm-halving replicates for OPT only: 42 changes applied, 16 helped, 11 harmed (vs. 30),
    WER 18.19 (CI vs. always [−0.90, +0.47]). For Jev the tuned threshold applies everything.
  - The oracle headroom is larger than on T15: 1.8 (OPT) and 2.0 (Jev) points, against a net gain of 0.
  - The blend weight is not robust on T12: Qwen's α is 0.4 tuned on a random split and 0.0 on the early
    sessions; OPT's 0.25 vs. 0.4 moves late WER by 0.41 points (P4's bar is 0.2).
  - Calibration degrades on later sessions for all three arms (P5), which T15 did not show.
  - Random split (secondary): the gate beats always-apply for all three arms, for Qwen significantly
    (−0.66, CI [−1.11, −0.19]). The gap between the two splits is the argument for temporal evaluation.
- **Training-text overlap audit** (required by CLAUDE.md, never run on T15 until now): evaluation sentences
  that also occur in the decoder's training text (`cina/data/train_sentences*.txt`, normalized). T15: 128 of
  1,252 (Cinà reports 107; his normalization differs), **117 of them in the late sessions (18% of 651)**.
  T12: 25, one of them late. `--exclude-train-overlap` drops them and writes `_nooverlap` results; the
  early/late session boundary stays the full data's (recomputing it on fewer sentences moved it 3
  sessions earlier, mixing two changes). Every result without the flag is unchanged.
- **Without overlap, the conclusions stand.** T15 late (534 sentences): the gate cuts harmful corrections
  for OPT (8 vs. 16) and Qwen (13 vs. 30) at equal or lower WER; Jev's gate applies everything, as before.
  T12 late: OPT 12 vs. 30, Jev 27 vs. 33. Across both participants and both overlap treatments, the
  harm-halving holds for OPT every time, for Qwen on T15, and never for Jev.
- **Next:** pooled T15+T12 (exploratory, labeled as such), then step 4's decode script.

## 2026-10-16 · Step 4 pipeline: built and tested before any list exists

- **The 3-gram needs no browser download.** `languageModel.tar.gz` (14.1 GB) on the Hugging Face mirror
  `yw784/brain-to-text-25-competition` has the SHA-256 Dryad publishes (`7b64d36c…6d7e`); the pod fetches
  and checks it. The archive unpacks to `./languageModel/` (`LG.fst` 9.4 GB first, then `TLG.fst`, `words.txt`).
- **Split of work, by cost** (RunPod balance $6.84): logits on the Mac (`src/step4_logits.py`, idle jobs
  `de6f33`, `5938d8`; the script recomputes step 2's greedy PER per session and refuses to write on any
  mismatch); decoding and scoring on one pod (`cloud/pod_decode.sh`): the published decoder built as in
  `setup_lm.sh` (Python 3.9, NumPy 1.24.4, libtorch 1.6 fetched by its CMake), Cinà's `build_lists_kaldi.py`
  and `runscores_cuda.py` unchanged, so scores match his (bf16, CUDA, per-candidate log-likelihood). A host
  with ≥80 GB RAM (`minRAMPerGPU`), 24 GB+ GPU.
- **Launcher:** `cloud/runpod.py run --job decode` reuses the tested train-job paths (cleanup on every
  failure, retried delete, self-stop, wake). Pod steps log to `out/logs/<step>.log`; the monitored log gets
  only milestones and a 5-minute heartbeat with elapsed time (the build and a decode can be silent for more
  than the monitor's 30-minute stall limit), and each step has its own time limit on the pod.
- **Tests:** `cloud/test_runpod.py` 13 (3 new: decode job asks for 80 GB and starts the right script; a
  decode failure wakes with the failure; the train job is unchanged); `cloud/test_pod_decode.py` 4, run on the
  real `step()` extracted from the script (a failing command stops the step and the script; a hang is killed
  at its limit together with its child; heartbeats carry no step output). The hang test fails when the
  child kill is removed. The analysis (`src/step4_rescoring.py`) run with Cinà's released lists standing in
  for both pod outputs gives E0 difference 0.00 and churn ratio 1.00, and the 729/651 early/late split of
  step 1; the word-alignment counter passes six hand-made cases.
- **Readings fixed now, before the lists exist:** R2 and R5 do not name a rescorer, so each must hold for
  both OPT and Qwen; R1 and R4 are OPT as written. The gate's agreement feature counts OPT and Qwen only,
  since Jev is a hosted API outside this pipeline. The released-model reference for R1/R2 is Cinà's lists and
  scores with α refit on the same 21 early sessions.
- **Logits written** (idle jobs `de6f33`, 5 min; `5938d8`, 3 min): the early model on all 41 sessions
  (1,426 trials, 26 MB) and the released model on the 20 late ones (13 MB), in
  `/Volumes/Ugreen/Projects/rescoring-drift/step4/`. Greedy PER from the saved logits equals step 2's in
  every session (41/41 and 20/20), edit for edit.
- **First smoke launch never reached a pod:** the create endpoint rejected "NVIDIA RTX PRO 4500 Blackwell
  Server Edition", a name the GraphQL price list carries but the REST enum lacks. Dropped; a test now checks
  every GPU name against the enum (it fails with the old list). Nothing billed.
- **Smoke run 1** (RTX 3090, $0.17, pod deleted): the decoder **built in 6 min** and the 3-gram was fetched,
  SHA-256-checked and unpacked in 9 min. The decode failed at once: `ModuleNotFoundError: No module named
  'numpy._core'`. The logits were pickled under NumPy 2, and the decoder's environment pins NumPy 1.24
  (`setup_lm.sh`). `src/step4_logits.py` now pickles arrays through `numpy.ndarray(shape, dtype, buffer)`;
  the two files were rewritten (round trip identical under NumPy 2.5). In a local Python 3.9 + NumPy 1.24.4
  venv the old file fails with the pod's exact error and the new ones load.
- **Scorer checked locally** with the pod's pins (transformers 4.56.2, accelerate 1.10.1): Cinà's
  `runscores_cuda.py` with `"cuda"` replaced by `"cpu"` in a copy, distilgpt2, two toy lists. Runs.
- **Found in Cinà's scorer:** `ARScoreRescorer` scores tokens 2..n of each candidate. OPT's tokenizer
  prepends `</s>`, so every word is scored; Qwen2.5's prepends nothing, so **the first word of every Qwen
  candidate is never scored**, and a one-token candidate gets 0.0, the maximum (seen with distilgpt2 on
  "hello"). No candidate in the published T15 lists gets a 0.0, so his Qwen results are not hit by the
  one-token case, but the first-word gap applies to all of them. Kept unchanged here so scores stay
  comparable with his; the analysis counts lists where a 0.0 wins. → ARTICLE-NOTES.
- **Next:** smoke run 2, then the full run.
- **Smoke run 2 passed end to end** (RTX 3090, ~$0.20, pod deleted): build 5 min, 3-gram fetch + check +
  unpack 7 min, both 1-gram decodes of 20 trials under a minute each (early-model lists: truth in list 25%,
  oracle 17.6%; released: 55%, 9.2%), scorer install 1 min, 3 lists scored per model per file (median
  ~0.3 s per list). Model snapshots: Qwen2.5-7B-Instruct `a09a354`; OPT-6.7b `a45aa65` (as in step 3b) plus
  `3f007b4`, which transformers also fetched. The fetched lists and scores load through the analysis's
  builtins-only unpickler.
- **Full run:** RTX 3090 only, `--hard-stop 6h`: worst case $3 against a $6.33 balance.

## 2026-10-16 · Step 4: the full run, E0 and R1–R5

- **Run:** `cloud/runpod.py run --job decode --name step4 --hard-stop 6h`, pod `jkvld0q3wel063` (RTX 3090),
  ~$0.46, deleted by the launcher; every pod step ok (`runs/step4-cuda/train.log`). Build 5 min, 3-gram
  6 min, decodes 15 min (early, 1,426 trials) and 7 min (released, late), scoring 2–5 min per file and model.
  Step 4 cost about $0.83 in all with the two smoke runs. Model snapshots as in smoke run 2.
- **Lists:** early model, 1,397 sentences, oracle WER 30.0%, truth in list 54%, mean list 60; released model
  on the late sessions, 668 sentences, oracle 2.5%, truth in list 89%, mean list 40.
- **Analysis:** `python -I src/step4_rescoring.py` → `results/step4_rescoring.json`; with
  `--exclude-train-overlap` → `results/step4_rescoring_nooverlap.json` (flag added in this session; it drops
  11 fit-set sentences and 117 far-regime ones, none from the near regime).
- **E0 holds:** 7.82% vs. Cinà's 7.75% first-pass WER on 651 late sentences (no overlap: 8.31 vs. 8.28).
- **R1 holds:** near-regime churn 0.34 (OPT) / 0.37 (Qwen) on the drifted decoder vs. 0.07 / 0.11 released.
- **R2 fails:** near first pass 23.8% → OPT 22.2, Qwen 22.0 (gains 1.64, 1.88; 6.9% and 7.9% of first-pass
  WER). Released on the same sentences: 3.28 → OPT 3.21, Qwen 3.44 (+0.08, −0.16). The drifted decoder gains
  more points and a larger share, so "more to fix, smaller share" is wrong: the share grows.
- **R3 holds:** far oracle 76.8% (near 11.3%, fit sessions 2.5%). Far first pass 89.7% → OPT 89.4%.
- **R4 fails:** OPT fixed/broken 45/24 = 1.88 near, 39/30 = 1.30 far. Without overlap 51/27 = 1.89 near,
  40/20 = 2.00 far, which holds.
- **R5 fails (Qwen):** near, gate vs. always: OPT 22.36 vs. 22.20, 59 vs. 65 changes, harmed 14 vs. 15;
  Qwen 22.60 vs. 21.97, 36 vs. 69 changes, harmed 7 vs. 16, helped 19 vs. 31; Qwen's 95% interval vs.
  always +0.09 to +1.00. Without overlap (OPT α 0.45 → 0.50): OPT 22.20 vs. 21.97, Qwen 22.36 vs. 21.97
  with 4 harmed, which holds. The near test set is identical in both runs; 11 fewer fit sentences moved the
  gate by 0.24 points.
- **`zero_score_wins`:** Qwen's 0.0 one-token score wins 4 lists, all in the far regime; none for OPT.
  The first-word gap does not move any verdict.
- **Reading:** while the lists still contain the sentence (near), the rescorer helps the drifted decoder
  more than the calibrated one, and the gate that halved harm for free on calibrated sessions (step 3a)
  now pays for it in lost fixes: always-apply is the better policy. Once the lists lose the sentence (far),
  nothing downstream matters; the fix is recalibration.
- **Next:** pooled T15+T12 (exploratory), then the write-up summary.

## 2026-10-17 — paper draft

Drafted `paper/draft.md` (about 3,400 words of body, every results paragraph followed by an HTML comment
naming its JSON source) and `paper/TODO-PUBLIC-REPO.md` (16 hardcoded SSD paths, `reproduce.py` spec,
Dryad digests, license, seven figures with their inputs). Two SUMMARY.md sentences corrected while checking
numbers: timestamps start at step 3a (a8486e0), and the 85/63 word counts are the random-split test set;
the later-session figures are 57 helped / 34 harmed sentences (`results/step3a_gate.json`). New observation:
`step4_rescoring_nooverlap.json` has 13 far-regime sessions, because t15.2025.03.14 and 2025.03.16 consist
entirely of training-text sentences. Committed on branch `paper-draft`; publication decided separately.

## 2026-10-18 · Release preparation: portable paths, reproduce.py, figures, corrections

- **Portable paths.** Every hardcoded data path now comes from `src/paths.py`: `RESCORING_DRIFT_DATA`
  (default `./data`) holds Cinà's and the official clones, the Dryad files, runs and activations. The step 4
  pod outputs (lists and scores, 8.2 MB, builtins-only pickles) are committed in `derived/step4/` so step 4
  reruns from a clone. `cloud/runpod.py` reads `RUNPOD_API_KEY` before the keychain and reports through
  `remind` only when that command exists; its 20 tests pass. The `checkpoint` and `output` fields in
  `results/step2_per_*.json` and `results/step3b_activations_meta.json` keep the original paths as the record.
- **`reproduce.py`**: 294 checks of the paper's numbers against `results/*.json` and `derived/step4/`, exit
  non-zero on any mismatch. `--rerun` (with `RESCORING_DRIFT_DATA=/Volumes/Ugreen/Projects/rescoring-drift`)
  re-executed steps 1, 3a (T15, T12, with and without overlap) and 4 in about 4 minutes: **17/17 result files
  byte-identical**, 294/294 checks pass.
- **Corrections found while writing the checks** (the earlier entries above stay as written):
  - T12 random split, Qwen gate − always is **−0.65** (21.81 − 22.47 = −0.6516), not −0.66 (entry 2026-10-14).
  - The overlap audit found 128 of **1,380** T15 sentences with a unique session; 1,252 is the count left
    after excluding them (entry 2026-10-14).
  - Qwen's near-regime gate cost moves **0.23** points between the full and overlap-excluded runs
    (0.6265 − 0.3909), not 0.24 (entry 2026-10-16, which subtracted rounded values).
  - The early model's PER on the sessions from day 411 to 504 lies between **73.8% and 81.1%**; SUMMARY.md and
    the step 2 outcome in PREDICTIONS.md said 75–80%.
  - Rescoring's net on the held-out sessions ranges from **−0.21** (Jev on T12) to +0.85 points; the draft
    said "between zero and 0.85".
  - The relative-gain correlations in step 1 (ρ = −0.00, 0.20, −0.12) are over the 37 sessions with ≥ 10
    sentences and any first-pass error; they are not stored in a result file and `reproduce.py` recomputes them.
- **New number:** the drifted decoder's 100-best lists contain the reference for 6 of 462 far-regime sentences
  and 115 of 189 near-regime ones (computed from `derived/step4/lists_early.pkl`).
- **Identity edits in this file:** names and private tool paths in earlier entries were replaced with neutral
  wording before publication; no number, command or result changed.
- Full digests: T12 `competitionData.tar.gz` =
  `5b5c9784d80d5286e69352621366be2e4ff468e043e082e7aa7cbc90d18336b3`; `t15_pretrained_rnn_baseline.zip` =
  `04ab2275eeb48abad9b4011dde6f7c9e5dd9f643b8a677c1ba5e06dc53bb4111` (computed locally from the files that
  matched Dryad's listing on 2026-10-08 and 2026-10-14).
- Paper: `paper/paper.md` (renamed from `draft.md`), three figures from `paper/make_figures.py`, PDF from
  `paper/build_pdf.sh` (pandoc 3.11 + typst 0.15.1).
- **Timestamp upgraded:** `timestamps/PREDICTIONS-1b0cd42.md.ots` was still pending only; `ots upgrade` today
  completed it (earliest attestation Bitcoin block 970724). Earliest blocks of the others: a8486e0 970412,
  41b8dff 971147, a204603 971469 (`ots info`). Each proof keeps its other attestations.

## 2026-10-25 · Pre-submission review and revision

A read-only review of the manuscript by a second model (Codex CLI, `codex exec --sandbox read-only`), 32 items,
is kept outside the repository with the point-by-point response. Every data claim in it that could be checked
was checked, and `src/review_checks.py` now computes the numbers the revision cites, into
`results/review_checks.json`, which `reproduce.py` checks (307 checks).

- The training-text overlap list is the released model's. Of the 11 fit-set and 117 far-regime sentences it
  removes in the drift analyses, 1 and 0 occur in the early model's own training sessions (the 4,175 distinct
  training sentences up to 2023.11.26). §2.6 and §3.5 now say so, and the exclusion is reported for the drifted
  decoder as a sensitivity to removing a prompt cohort.
- P1 was registered word-weighted; step 1 computed session-mean quartiles. Word-weighted: 17.7% against 1.8%,
  10.0-fold (session means 17.6% against 1.5%, 11.4-fold). Both are in §3.1 and the deviation is in §2.7.
- Of the 4 far-regime lists whose top Qwen score is 0.0, 1 selects that candidate at the tuned α.
- The 17 session-ambiguous sentences fall 5 in Cinà's 419-sentence development split and 12 in the test split;
  §3.2 and §4 had said 17 of 419.
- Prose changes: the scoring equation in §2.3, the confidence feature defined in §2.4, a design table (Table 1)
  and a prediction/verdict table (Table 2), scoped claims for the probe, the drift mechanism, the gate's
  failure under drift and the "only fix" sentence, and six references added (one DOI from the review was
  wrong: Wessel et al. 2001 is 10.1109/89.906002).

## 2026-10-25 · Step 5: the four controls the review asked for

Predictions S1–S4 committed first (`PREDICTIONS.md`), then `src/step5_review.py` (CPU, ~2 min: near-regime
leave-one-session-out, the random-feature control for the probe over 20 seeds, feature distributions and
matched list lengths) and `src/step5_input_layer.py` (released checkpoint through the frozen 2023.11.26 input
layer on the 20 late sessions, MPS, ~5 min). Results in `results/step5_review.json` and
`results/step5_input_layer.json`; `reproduce.py` checks them (349 checks).

- S1: OPT's near-regime verdict rests on nothing (per session −0.47 to +0.60, leave-one-out 0.00 to +0.28);
  Qwen's +0.63 cost survives every leave-one-out (+0.42 to +0.76).
- S2: frozen input layer 45.6% late PER against 11.2% (own layers) and 56.3% (early model): the input layer
  carries about three quarters of the gap. Same shape, flat for two weeks then a jump at day 21.
- S3: random features cost a median +0.09 (0.00 to +0.25 over seeds); the real probe's +0.11 is inside.
  The probe is indistinguishable from noise at 90 fit examples.
- S4: agreement separates helpful from harmful by +0.76 sd calibrated and still +0.49/+0.56 sd in the near
  regime, so the feature keeps its signal; the refit gate gives it ~0 weight because on the drifted decoder's
  own training sessions it did not predict help. Matched list lengths: 88.5 (early model) against 39.2
  (released) on the same 20 sessions; the "60 against 40" in the first draft mixed populations.
- Paper: new §3.6, the account and limitations in §4 revised accordingly, Table 2 gets the S row.

## 2026-10-28 · Release: Zenodo DOIs reserved

- Two Zenodo drafts, DOIs reserved before publication so they can be written into the files: the preprint,
  doi:10.5281/zenodo.23745489 (PDF), and the data, doi:10.5281/zenodo.23745520 (the step 2 checkpoint with its
  configuration, training log and trial list, and the step 4 logits `logits_early.pkl` and
  `logits_released_late.pkl`, with `SHA256SUMS`). The DOIs are in the paper's title block and §6, the README
  and `CITATION.cff`.
- The paper and README said `reproduce.py` makes 294 checks; it has made 349 since step 5. Corrected.
- PDF rebuilt (`paper/build_pdf.sh`); `reproduce.py` 349/349 on this tree.
- The public repository starts from one commit of this tree; the private working history stays private. The
  OpenTimestamps proofs stamp file contents, so they verify without it.
