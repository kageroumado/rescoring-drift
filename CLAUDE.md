# rescoring-drift

Does a speech neuroprosthesis's language-model rescoring stage help or harm more as the neural decoder
drifts across sessions, and can a gate trained on early sessions decide, on later ones, when to correct,
keep the decoder's output, or ask for confirmation?

Paper: `paper/paper.md`. Check every number in it with `python3 reproduce.py`.

## Document as you go

The record matters as much as the code.

- **`LAB-NOTES.md`**: a dated entry for every session of work. Record what ran (the exact command), what came
  out (numbers, paths to result files), what broke and how it was fixed, and what to do next. Write the entry
  while working, not afterwards. Failed attempts and dead ends get entries too.
- **`ARTICLE-NOTES.md`**: anything a reader of the eventual write-up would find interesting. That covers
  surprising observations, numbers that contradict a published claim, methodological traps (leakage,
  overlap, alignment), figures worth making, and sentences that state a finding well. Each item links to
  the lab-notes entry it came from.
- **`PREDICTIONS.md`**: predictions are written and committed **before** the analysis that tests them
  runs. Give that commit an outside timestamp before running it: while the repo is private, copy the
  committed file to `timestamps/PREDICTIONS-<commit>.md` and run `ots stamp` on it (OpenTimestamps anchors
  only the hash in Bitcoin, so nothing is revealed); `ots upgrade` the `.ots` a few hours later and commit
  it. Software Heritage save-code-now archives public commits the same way. Git author dates alone can be set by hand.

## Evaluation rules

- Every fitted choice (thresholds, temperatures, probe weights, layer selection, acoustic weights) is fit on
  training sessions only. Report the random split only next to a later-sessions-held-out split.
- All candidates from one trial stay on the same side of a split.
- Audit overlap between evaluation sentences and the RNN's training text. Cinà found 107 overlapping
  sentences; report results with and without them.
- Word-level claims (helpful vs. harmful corrections) use explicit alignments, because insertions and
  deletions shift word positions.
- A null result gets the same write-up as a positive one.

## Data and third-party code

Large files live outside the repository, under `$RESCORING_DRIFT_DATA` (see `src/paths.py`).

| Path | What |
|---|---|
| `dryad/` | Card et al. (NEJM 2024) T15 data and pretrained RNN, CC0. Downloaded from a Hugging Face mirror; each file's SHA-256 matches the hash Dryad publishes (Dryad's own downloads need a login) |
| `cina/` | Clone of [gabrycina/how-much-language-model](https://github.com/gabrycina/how-much-language-model) (MIT): cached 100-candidate lists and per-rescorer scores |
| `nejm/` | Clone of [Neuroprosthetics-Lab/nejm-brain-to-text](https://github.com/Neuroprosthetics-Lab/nejm-brain-to-text): RNN code and data loaders |

Third-party pickles are audited before loading: list their GLOBAL/STACK_GLOBAL opcodes with `pickletools`
and load only files that import nothing beyond builtins. Run Python that reads downloaded files with `-I`.

## Conventions

- `src/` holds the scripts; each writes its outputs to `results/` (small JSON/CSV, committed) or to
  `$RESCORING_DRIFT_DATA` (anything large).
- Cite Cinà's work wherever this builds on his lists, scores or findings.
- Never `rm`; use `trash`.
