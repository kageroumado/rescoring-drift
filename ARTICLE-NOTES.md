# Article notes

Material for the eventual write-up. Each item points to its lab-notes entry.

- **The benchmark's official download path has decayed.** Dryad now requires a login for API and scripted
  downloads, so the official `download_data.py` fails. Mirrors exist, but only a published checksum makes
  them trustworthy. A small reproducibility point for a field that relies on one public dataset.
  (Lab notes, 2026-10-08)
- **The published evaluation never asks "later than training".** Cinà's abstention curves and tuning use
  a random split over sentences, mixing sessions from all 20 months. A deployed decoder only ever meets
  sessions after its calibration. (Lab notes, 2026-10-08)
- **The rescoring stage's small net gain hides large churn.** Qwen2.5-7B fixes 85 words and breaks 63 on the
  same test set (net 22); Jev fixes 121 and breaks 81. A perfect "correct or keep" gate would cut WER by
  0.5–1.3 points, more than any rescorer's net gain. The deployment question is selective correction, not
  a better rescorer. (Lab notes, 2026-10-08, step 1)
- **Sessions differ 11× in difficulty; the rescorer doesn't care.** The hardest quartile of sessions
  averages 17.6% first-pass WER, the easiest 1.5%, yet every rescorer removes about the same share of errors
  in both. "Rescoring matters more for a weak decoder" is, across natural sessions, just "a weak decoder has
  more errors". (Lab notes, 2026-10-08, step 1)
- **Text-keyed caches silently drop repeated sentences.** 17 sentences were spoken in several sessions;
  keying candidate lists by reference text keeps one list for 45 trials. A trial-keyed cache would have
  kept them. (Lab notes, 2026-10-08, step 1)
- **A gate can halve the language model's harmful edits without changing WER.** On later sessions, the
  output-feature gate cuts Qwen2.5-7B's harmful corrections from 34 to 17 sentences while keeping 49 of its 57
  helpful ones; WER moves 7.08 → 6.90. WER counts a word the model inserted the same as a word the decoder
  missed. For a person speaking through the device they are different failures. (Lab notes, step 3a)
- **The rescoring weight is fragile.** Dropping 17 of 419 tuning sentences moves OPT-6.7b's best blend
  weight from 0.3 to 0.5, which doubles the sentences it changes and erases its net gain. (Lab notes, step 3a)
- **A default argument nearly skewed a calibration result.** Cinà's temperature fit silently uses acoustic
  scale 0.35 while the decoder runs at 0.3. It changed nothing here, but this is the kind of mismatch a
  reproduction should name. (Lab notes, step 3a)
- **The rescorer's activations don't know when it is wrong.** A probe on OPT-6.7b's residual stream, added to
  the output-feature gate, makes later-session WER worse by 0.11 points (95% interval +0.04 to +0.20) and
  leaves discrimination flat (AUC 0.63 → 0.64). A clean negative for the "look inside the model" idea at
  this scale: the useful signal is in agreement between models and score margins, not in one model's
  hidden state. (Lab notes, step 3b)
- **Reproducing a cached score is a cheap sanity anchor.** Rescoring 30 lists locally (float16, Apple GPU)
  matched Cinà's CUDA bfloat16 scores at Spearman 0.9998, which licenses using local activations against
  his cached decisions. (Lab notes, step 3b)

## Drift is immediate, not gradual (LAB-NOTES 2026-10-13)

- A decoder trained through 2023.11.26 triples its phoneme error rate one week later (14.8% vs. 4.8% for
  the recalibrated model) and passes 60% within ~100 days. The day-to-day input-layer change, not slow decay,
  carries most of the damage: day 7 to day 14 is flat, day 21 jumps to 36%.
- Figure: per-session PER vs. days since the last training session, early model vs. released, log x-axis.
- Consequence for the paper's framing: "the rescorer under drift" has a narrow window where the question is
  live; after that no rescorer can help, and the honest answer is "recalibrate".

- **On a second participant, rescoring nets exactly zero** (T12 late sessions: 18.40 → 18.40 with OPT,
  33 sentences helped, 30 harmed), while the oracle headroom is 1.8 points. The LM stage is a coin flip
  that a good gate could turn into a gain. → LAB-NOTES 2026-10-14
- **A random split flatters the gate; a temporal one doesn't.** Same T12 data: the gate wins for all three
  rescorers on a random split, for one on a temporal split. → LAB-NOTES 2026-10-14
- **The Qwen rescorer never reads the first word.** Per-candidate log-likelihood scores tokens 2..n; OPT's
  tokenizer adds a start token, Qwen2.5's adds none, so Qwen judges each candidate without its first
  word, and a one-token candidate scores a perfect 0.0. A trap for anyone comparing LM rescorers across
  tokenizers. → LAB-NOTES 2026-10-16
- **Under drift the rescorer matters more, not less, and the gate stops paying.** A week to a month after
  calibration, first-pass WER is 23.8% instead of 3.3%; OPT cuts 6.9% of the errors (released: 2.4%), and
  Qwen, which hurts the calibrated decoder, helps the drifted one by 1.9 points. The gate that halved Qwen's
  harmful edits for free on calibrated sessions still halves them (7 vs. 16) but drops 12 of 31 helpful
  ones, costing 0.6 points. → LAB-NOTES 2026-10-16 (step 4)
- **Past ~3 months no rescorer can help:** the best of 100 candidates has 77% WER; rescoring moves 89.7% to
  89.4%. Figure: oracle and first-pass WER by regime, drifted vs. released. → LAB-NOTES 2026-10-16
- **Pre-registration caught a fragile verdict.** Removing 11 fit sentences that overlap the decoder's
  training text flips two step 4 verdicts, with the near-regime test set unchanged. Reporting the
  registered run first and the sensitivity second is the only honest order. → LAB-NOTES 2026-10-16
