"""Step 4: rescoring and the gate on a drifted decoder (predictions E0, R1-R5 in PREDICTIONS.md).

Inputs are the pod's outputs (cloud/pod_decode.sh): 100-candidate lists from the step 2 early model's logits
on all sessions (`lists_early.pkl`) and from the released model's logits on the late sessions
(`lists_released_late.pkl`), decoded with the published Kaldi/WFST decoder and OpenWebText 3-gram, plus
OPT-6.7b and Qwen2.5-7B per-candidate scores for both. The reference is Cinà's lists and scores from the
released model.

Every fitted choice (blend weight α, softmax temperature, gate weights and threshold) is fit on the 21
sessions up to 2023.11.26 and evaluated on later ones. Regimes, fixed in PREDICTIONS.md from step 2's PER:
near = late sessions within 33 days of training, far = from day 91 on.

    python src/step4_rescoring.py [--results <dir with the pod's out/ files>] [--exclude-train-overlap]
"""

import argparse
import json
import sys
from pathlib import Path

import jiwer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step1_sessions as S  # noqa: E402
import step3a_gate as G  # noqa: E402

A = S.A
ARMS = ["opt", "qwen"]  # Jev is a hosted API outside this pipeline; the gate's agreement feature counts these two
from paths import STEP4_POD_OUTPUTS as RESULTS_DIR  # noqa: E402

NEAR_MAX_DAYS, FAR_MIN_DAYS = 33, 91
E0_BAR, R1_RATIO, R3_BAR, R5_BAR = 0.5, 2.0, 40.0, 0.5


def load_lists(results: Path, which: str):
    cache, meta = S.read_pickle(results / f"lists_{which}.pkl")
    scores = {}
    for arm in ARMS:
        raw = S.read_pickle(results / f"scores_{which}_{arm}.pkl")["scores"]
        scores[arm] = {s: np.asarray(raw[s]) if s in raw else np.zeros(len(cache[s])) for s in cache}
    return cache, meta, scores


def regimes():
    """Session -> 'train', 'near', 'far' or 'between' from step 2's days after training."""
    rows = json.loads((S.RESULTS / "step2_per_early.json").read_text())["sessions"]
    out = {}
    for r in rows:
        days = r["days_after_training"]
        out[r["session"]] = ("train" if r["trained_on"] else "near" if days <= NEAR_MAX_DAYS
                             else "far" if days >= FAR_MIN_DAYS else "between")
    return out


def keys_by_regime(session_of, regime_of):
    out = {"train": [], "near": [], "far": [], "between": []}
    for s, sess in session_of.items():
        out[regime_of[sess]].append(s)
    return out


def wer(err, picks, keys):
    return S.wer(err, picks, keys)


def oracle_wer(err, cache, keys):
    words = sum(len(s.split()) for s in keys)
    return 100 * sum(min(err(s, h) for h, _, _ in cache[s]) for s in keys) / words


def word_changes(ref: str, before: str, after: str) -> tuple[int, int]:
    """Words fixed and broken by replacing `before` with `after`, from explicit alignments to the reference:
    a reference word is correct when it sits in an 'equal' chunk; inserted words count as errors."""
    def correct_and_insertions(hyp):
        out = jiwer.process_words(ref, hyp)
        correct = set()
        for chunk in out.alignments[0]:
            if chunk.type == "equal":
                correct.update(range(chunk.ref_start_idx, chunk.ref_end_idx))
        return correct, out.insertions
    c0, i0 = correct_and_insertions(before)
    c1, i1 = correct_and_insertions(after)
    fixed = len(c1 - c0) + max(0, i0 - i1)
    broken = len(c0 - c1) + max(0, i1 - i0)
    return fixed, broken


def always_apply(err, cache, scores, train, keys, arm):
    """α fit on `train`; first-pass and rescored WER, churn and word-aligned fixes on `keys`."""
    alpha = S.tune_alpha(err, cache, scores[arm], train)
    first = A.picks(cache, None, 0.0, S.ACOUSTIC_SCALE)
    picks = A.picks(cache, scores[arm], alpha, S.ACOUSTIC_SCALE)
    changed = [s for s in keys if picks[s] != first[s]]
    fixed = broken = 0
    for s in changed:
        f, b = word_changes(s, first[s], picks[s])
        fixed, broken = fixed + f, broken + b
    w_first, w_always = wer(err, first, keys), wer(err, picks, keys)
    return {"alpha": alpha, "sentences": len(keys), "first_pass_wer": w_first, "always_wer": w_always,
            "gain": w_first - w_always, "relative_gain": (w_first - w_always) / w_first if w_first else None,
            "churn": len(changed) / len(keys) if keys else None, "changed": len(changed),
            "words_fixed": fixed, "words_broken": broken,
            "fixed_to_broken": fixed / broken if broken else float("inf")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=RESULTS_DIR)
    ap.add_argument("--exclude-train-overlap", action="store_true")
    a = ap.parse_args()
    S.configure("t15", a.exclude_train_overlap)
    S.ARMS = ARMS
    err = S.Errors()
    regime_of = regimes()

    early_cache, early_meta, early_scores = load_lists(a.results, "early")
    rel_cache, rel_meta, _ = load_lists(a.results, "released_late")
    cina_cache, cina_scores = S.load()
    early_map, early_report = S.session_map(early_cache)
    cina_map, _ = S.session_map(cina_cache)
    rel_map, _ = S.session_map(rel_cache)
    ours = keys_by_regime(early_map, regime_of)
    cina = keys_by_regime(cina_map, regime_of)
    out = {"lists": {"early": early_meta, "released_late": rel_meta}, "mapping_early": early_report,
           "sessions": {r: sorted({early_map[s] for s in ks}) for r, ks in ours.items()},
           "sentences": {r: len(ks) for r, ks in ours.items()},
           # Cinà's per-candidate scorer skips the first token; Qwen adds no start token, so a one-token
           # candidate scores 0.0, the maximum. Lists where such a candidate is the top score, per regime:
           "zero_score_wins": {arm: {r: sum(max(early_scores[arm][s]) == 0.0 for s in ks) for r, ks in ours.items()}
                               for arm in ARMS}}

    # E0: the same pipeline on the released model's logits against Cinà's lists, late sessions.
    late = [s for s in rel_map if regime_of[rel_map[s]] != "train" and s in cina_cache]
    first_rel = A.picks(rel_cache, None, 0.0, S.ACOUSTIC_SCALE)
    first_cina = A.picks(cina_cache, None, 0.0, S.ACOUSTIC_SCALE)
    e0 = {"sentences": len(late), "ours": wer(err, first_rel, late), "cina": wer(err, first_cina, late)}
    e0["difference"] = e0["ours"] - e0["cina"]
    e0["holds"] = abs(e0["difference"]) <= E0_BAR
    out["E0"] = e0

    # R1, R2: near regime, early model against the released model on the same trials.
    near = [s for s in ours["near"] if s in cina_cache]
    out["R1_R2"] = {}
    for arm in ARMS:
        drifted = always_apply(err, early_cache, early_scores, ours["train"], near, arm)
        released = always_apply(err, cina_cache, cina_scores, cina["train"], near, arm)
        out["R1_R2"][arm] = {"early_model": drifted, "released_model": released,
                             "churn_ratio": drifted["churn"] / released["churn"] if released["churn"] else None}
    r = out["R1_R2"]
    out["R1_holds"] = r["opt"]["churn_ratio"] is not None and r["opt"]["churn_ratio"] >= R1_RATIO
    out["R2_holds"] = all(r[arm]["early_model"]["gain"] > r[arm]["released_model"]["gain"]
                          and r[arm]["early_model"]["relative_gain"] < r[arm]["released_model"]["relative_gain"]
                          for arm in ARMS)

    # R3: the far regime's lists no longer hold the sentence.
    out["R3"] = {"far_oracle_wer": oracle_wer(err, early_cache, ours["far"]),
                 "near_oracle_wer": oracle_wer(err, early_cache, ours["near"]),
                 "train_oracle_wer": oracle_wer(err, early_cache, ours["train"])}
    out["R3_holds"] = out["R3"]["far_oracle_wer"] > R3_BAR

    # R4: word-aligned fixes per break for always-apply OPT, far vs. near.
    out["R4"] = {reg: always_apply(err, early_cache, early_scores, ours["train"], ours[reg], "opt")
                 for reg in ("near", "far")}
    out["R4_holds"] = out["R4"]["far"]["fixed_to_broken"] > out["R4"]["near"]["fixed_to_broken"]

    # R5: the step 3a gate, fit on the early sessions, on each regime.
    out["R5"] = {}
    for reg in ("near", "far"):
        setting = G.Setting(err, early_cache, early_scores, ours["train"], ours[reg])
        out["R5"][reg] = G.run(setting, early_map)
    near_arms = out["R5"]["near"]["arms"]
    out["R5_holds"] = all(
        "wer" in near_arms[arm]
        and near_arms[arm]["applied"]["logistic_gate"]["harmed"] < near_arms[arm]["applied"]["always"]["harmed"]
        and near_arms[arm]["wer"]["logistic_gate"] <= near_arms[arm]["wer"]["always"] + R5_BAR
        for arm in ARMS)

    S.result_path("step4_rescoring").write_text(json.dumps(out, indent=1, default=float))
    summary = {k: out[k] for k in ("sentences", "zero_score_wins", "E0", "R1_holds", "R2_holds", "R3", "R3_holds", "R4_holds",
                                   "R5_holds")}
    print(json.dumps(summary, indent=1, default=float))
    for arm in ARMS:
        d, rel = out["R1_R2"][arm]["early_model"], out["R1_R2"][arm]["released_model"]
        print(f"{arm} near: churn {d['churn']:.2f} vs {rel['churn']:.2f} | first {d['first_pass_wer']:.1f} -> "
              f"{d['always_wer']:.1f} vs {rel['first_pass_wer']:.1f} -> {rel['always_wer']:.1f}")
    for reg in ("near", "far"):
        q = out["R4"][reg]
        print(f"opt {reg}: fixed {q['words_fixed']} broken {q['words_broken']} ratio {q['fixed_to_broken']:.2f}")
        for arm, g in out["R5"][reg]["arms"].items():
            if "wer" in g:
                print(f"  {arm} {reg}: always {g['wer']['always']:.2f} (-{g['applied']['always']['harmed']}) gate "
                      f"{g['wer']['logistic_gate']:.2f} (-{g['applied']['logistic_gate']['harmed']}) "
                      f"oracle {g['wer']['oracle']:.2f}")


if __name__ == "__main__":
    main()
