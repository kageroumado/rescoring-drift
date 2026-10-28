# Summary: the language model on a drifting speech decoder

A speech neuroprosthesis decodes phonemes from motor cortex and lets a language model choose among the
decoder's 100 best sentences. This project asks what that rescoring stage does as the neural decoder
drifts away from its calibration, and whether a gate fit on early sessions can tell, on later ones, when
to accept the language model's change. Data: Card et al. (NEJM 2024), participant T15, 45 sessions over
20 months; participant T12 for replication. It builds on Cinà's cached lists and rescorer scores
([how-much-language-model](https://github.com/gabrycina/how-much-language-model)). Every prediction was
committed, and from step 3a on timestamped, before the analysis that tested it (`PREDICTIONS.md`); the
record of every run is `LAB-NOTES.md`.

## Findings

1. **On a calibrated decoder, rescoring is churn with a small net.** On T15's random-split test set
   Qwen2.5-7B fixes 85 words and breaks 63 (on later sessions, 57 sentences helped and 34 harmed); on T12 OPT-6.7b nets exactly zero (33 sentences helped, 30 harmed). A
   perfect correct-or-keep gate would gain 0.5–2.0 points, more than any rescorer's net gain. (Step 1, T12)
2. **An output-feature gate halves harm on T15 and does not reliably replicate.** On T15 it cuts Qwen's
   harmful changes from 34 to 17 at no WER cost; on T12 the registered replication (H1) fails: it holds for
   OPT only. Agreement between rescorers leads the gate's coefficients for OPT on T15 and for both
   fittable arms on T12. (Steps 3a, T12)
3. **The rescorer's activations add nothing.** A probe on OPT-6.7b's residual stream makes the gate worse
   by 0.11 points (95% interval +0.04 to +0.20). (Step 3b)
4. **Drift is immediate.** A decoder trained on the first 21 sessions has 3× the recalibrated decoder's
   phoneme error rate one week later, 36% by day 21, 64% by day 103, and 74–81% after a year. (Step 2; D1–D3
   hold)
5. **Under drift the rescorer matters more, and the gate stops paying.** Within a month of calibration
   (first-pass WER 23.8% against 3.3% calibrated), the rescorer changes 5× as many sentences and removes a
   larger share of errors (OPT 6.9% vs. 2.4%; Qwen helps by 1.9 points where it hurts the calibrated
   decoder). The gate still vetoes harmful changes but now vetoes helpful ones too: always-apply wins.
   After three months the 100-best lists no longer hold the sentence (oracle WER 77%) and nothing
   downstream of the decoder helps; the fix is recalibration. (Step 4)

## Step 4 verdicts

| | Prediction | Verdict |
|---|---|---|
| E0 | our pipeline reproduces Cinà's first-pass WER within 0.5 | holds (7.82 vs. 7.75) |
| R1 | near-regime churn ≥ 2× the calibrated decoder's | holds (5.0×) |
| R2 | larger gain in points, smaller as a share | fails: the share grows too |
| R3 | far-regime oracle WER above 40% | holds (76.8%) |
| R4 | fixed-to-broken ratio higher far than near | fails (1.30 vs. 1.88); holds without training-text overlap |
| R5 | gate cuts harm within 0.5 points, both rescorers | fails for Qwen (+0.63); holds without overlap |

R4 and R5 depend on 11 fit sentences and 117 far-regime sentences that also occur in the decoder's
training text. The registered verdicts are the full-data ones; the sensitivity is part of the result.

## Exploratory: what the gate vetoes (not pre-registered)

Pooled over the arms with a fittable gate, the changes the gate refuses split as follows. On calibrated
decoders (T15 OPT and Qwen, T12 OPT): 107 vetoed, 46 of them harmful and 35 helpful. On the drifted
decoder, near regime (OPT and Qwen): 39 vetoed, 10 harmful and 15 helpful. The gate's features separate
good from bad changes on the sessions it was fit to and lose that separation once the decoder drifts:
score margins move with the decoder.

## Traps worth knowing

- Dryad's official download path needs a login; mirrors are usable only against its published checksums.
- Cinà's per-candidate scorer skips the first token. Qwen2.5 adds no start token, so its scores never read a
  candidate's first word, and a one-token candidate scores a perfect 0.0 (it wins 4 far-regime lists here).
- Caches keyed by sentence text keep one list for repeated sentences (45 trials collapse to 17 lists).
- The blend weight α is fragile: dropping 17 of 419 tuning sentences moves OPT's from 0.3 to 0.5.
- Cinà's metadata for the T12 lists names the source as T15.

## Cost

Rented GPU time: step 2 training and step 4 decoding and scoring on RunPod, including one idle incident
($12.57) recorded in `LAB-NOTES.md` 2026-10-10. Step 4 in all cost about $0.83 (two smoke runs and the
full run, RTX 3090).
