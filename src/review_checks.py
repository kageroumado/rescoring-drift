"""Numbers the paper cites that no step script writes: the registered word-weighted form of P1, how
much of the training-text overlap list falls inside the early model's own training sessions, how
often a zero-scored Qwen candidate is actually selected, and where the session-ambiguous sentences
sit in Cinà's random split. Writes results/review_checks.json, which reproduce.py checks.

Needs Cinà's repository and, for the early-model overlap count, the Dryad T15 training transcripts.

    python src/review_checks.py
"""
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step1_sessions as S  # noqa: E402
import step4_rescoring as R4  # noqa: E402
from paths import T15_DATA  # noqa: E402
from revision_analysis import split  # noqa: E402
from sessions import normalize  # noqa: E402

EARLY_CUTOFF = "2023.11.26"


def p1_both_ways(participant):
    """Session-mean quartile WER (as step 1 computed it) and word-weighted (as P1 was registered)."""
    rows = [r for r in json.loads((S.RESULTS / f"step1_sessions{S.PARTICIPANTS[participant]['suffix']}.json").read_text())["per_session"]
            if r["sentences"] >= S.MIN_SESSION_SENTENCES]
    w = np.array([r["first_pass_wer"] for r in rows])
    n = np.array([r["words"] for r in rows], float)
    q1, q3 = np.percentile(w, [25, 75])
    easy, hard = w <= q1, w >= q3
    return {"session_mean": {"easy": float(w[easy].mean()), "hard": float(w[hard].mean()),
                             "ratio": float(w[hard].mean() / w[easy].mean())},
            "word_weighted": {"easy": float(np.average(w[easy], weights=n[easy])),
                              "hard": float(np.average(w[hard], weights=n[hard])),
                              "ratio": float(np.average(w[hard], weights=n[hard]) / np.average(w[easy], weights=n[easy]))}}


def early_training_sentences():
    out = set()
    for path in sorted(glob.glob(str(T15_DATA / "t15.*" / "data_train.hdf5"))):
        session = Path(path).parent.name
        if session.split(".", 1)[1] > EARLY_CUTOFF:
            continue
        with h5py.File(path, "r") as f:
            for key in f.keys():
                label = f[key].attrs["sentence_label"]
                out.add(normalize(label.decode() if isinstance(label, bytes) else str(label)))
    return out


def main():
    S.configure("t15")
    S.ARMS = ["opt", "qwen"]
    out = {"P1": {"t15": p1_both_ways("t15")}}
    S.configure("t12")
    out["P1"]["t12"] = p1_both_ways("t12")
    S.configure("t15")

    cache, _, scores = R4.load_lists(R4.RESULTS_DIR, "early")
    session_of, _ = S.session_map(cache)
    keys = R4.keys_by_regime(session_of, R4.regimes())
    overlap = S.train_text()
    early = early_training_sentences()
    fit_overlap = [k for k in keys["train"] if k in overlap]
    late_overlap = [k for k in keys["near"] + keys["far"] if k in overlap]
    out["early_model_overlap"] = {
        "early_training_sentences": len(early),
        "fit_in_released_training_text": len(fit_overlap),
        "fit_also_in_early_training": sum(k in early for k in fit_overlap),
        "late_in_released_training_text": len(late_overlap),
        "late_also_in_early_training": sum(k in early for k in late_overlap),
    }

    step4 = json.loads((S.RESULTS / "step4_rescoring.json").read_text())
    alpha = step4["R5"]["far"]["alpha"]["qwen"]
    picks = S.A.picks(cache, scores["qwen"], alpha, S.ACOUSTIC_SCALE)
    far_zero = [k for k in keys["far"] if max(scores["qwen"][k]) == 0.0]
    selected = [k for k in far_zero if scores["qwen"][k][[h for h, _, _ in cache[k]].index(picks[k])] == 0.0]
    out["qwen_zero_score"] = {"far_lists_with_zero_top_score": len(far_zero), "selected_at_tuned_alpha": len(selected)}

    cina_cache, _ = S.load()
    sessions_of = defaultdict(set)
    for r in json.loads((S.RESULTS / "val_sessions.json").read_text()):
        sessions_of[r["key"]].add(r["session"])
    ambiguous = {k for k, v in sessions_of.items() if len(v) > 1 and k in cina_cache}
    dev, test = split(cina_cache)
    out["session_ambiguous_in_random_split"] = {"ambiguous": len(ambiguous), "in_dev": len(ambiguous & set(dev)),
                                                "in_test": len(ambiguous & set(test)), "dev": len(dev), "test": len(test)}
    path = S.RESULTS / "review_checks.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
