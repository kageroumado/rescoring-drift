"""Step 1: how the rescoring stage's value varies across recording sessions (predictions P1-P5).

Uses Cinà's cached candidate lists and rescorer scores, his candidate-selection rule (analyze.picks) and
his calibration definitions (revision_analysis.softmax_conf / fit_temperature), with each sentence mapped
to its session by src/sessions.py's output. T15 is the default; --participant t12 runs the same analysis on
Cinà's T12 lists and writes results with a _t12 suffix.

    python src/step1_sessions.py [--participant t12]
"""

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import jiwer
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import CINA  # noqa: E402

sys.path.insert(0, str(CINA / "src"))
import analyze as A  # noqa: E402
from revision_analysis import softmax_conf, split  # noqa: E402
from sessions import normalize  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
ARMS = ["opt", "qwen", "jev"]
ACOUSTIC_SCALE = 0.3
ALPHAS = [float(a) for a in np.round(np.arange(0, 1.001, 0.05), 2)]
MIN_SESSION_SENTENCES = 10
LATE_FRACTION = 0.5
PARTICIPANTS = {
    "t15": {"lists": "lists_published.pkl", "scores": "scores_published/kaldi_{arm}.pkl",
            "sessions": "val_sessions.json", "train_text": "train_sentences.txt", "suffix": ""},
    "t12": {"lists": "lists_t12.pkl", "scores": "scores_t12/t12_{arm}.pkl",
            "sessions": "t12_sessions.json", "train_text": "train_sentences_t12.txt", "suffix": "_t12"},
}
PARTICIPANT = PARTICIPANTS["t15"]
EXCLUDE_TRAIN_OVERLAP = False
SPLIT_SESSIONS = {}  # the full session map, which fixes the early/late boundary when overlap is excluded


def configure(name, exclude_train_overlap=False):
    """Select whose lists, scores and session map every function here reads, and whether evaluation
    sentences that also occur in the decoder's training text are dropped."""
    global PARTICIPANT, EXCLUDE_TRAIN_OVERLAP
    PARTICIPANT = PARTICIPANTS[name]
    EXCLUDE_TRAIN_OVERLAP = exclude_train_overlap


def participant_arg():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", choices=sorted(PARTICIPANTS), default="t15")
    ap.add_argument("--exclude-train-overlap", action="store_true")
    a = ap.parse_args()
    configure(a.participant, a.exclude_train_overlap)


def result_path(stem):
    overlap = "_nooverlap" if EXCLUDE_TRAIN_OVERLAP else ""
    return RESULTS / f"{stem}{PARTICIPANT['suffix']}{overlap}.json"


def train_text():
    with open(CINA / "data" / PARTICIPANT["train_text"]) as f:
        return {normalize(line) for line in f if line.strip()}


class BuiltinsOnlyUnpickler(pickle.Unpickler):
    """Cinà's caches hold only dicts, lists, tuples, strings and numbers; anything importable is refused."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refusing {module}.{name} in a third-party pickle")


def read_pickle(path):
    with open(path, "rb") as f:
        return BuiltinsOnlyUnpickler(f).load()


class Errors:
    """Word errors per (reference, hypothesis), memoized: jiwer is the slow part."""

    def __init__(self):
        self.memo = {}

    def __call__(self, ref, hyp):
        if (ref, hyp) not in self.memo:
            m = jiwer.process_words([ref], [hyp])
            self.memo[(ref, hyp)] = m.substitutions + m.deletions + m.insertions
        return self.memo[(ref, hyp)]


def load():
    cache, _ = read_pickle(CINA / "data" / PARTICIPANT["lists"])
    zero = {s: np.zeros(len(nb)) for s, nb in cache.items()}
    scores = {}
    for arm in ARMS:
        raw = read_pickle(CINA / "data" / PARTICIPANT["scores"].format(arm=arm))["scores"]
        scores[arm] = {s: raw.get(s, zero[s]) for s in cache}
    return cache, scores


def session_map(cache):
    rows = json.loads((RESULTS / PARTICIPANT["sessions"]).read_text())
    sessions_of = defaultdict(set)
    for r in rows:
        sessions_of[r["key"]].add(r["session"])
    unique = {k: next(iter(v)) for k, v in sessions_of.items() if len(v) == 1 and k in cache}
    overlap = train_text() & set(unique)
    SPLIT_SESSIONS.clear()
    SPLIT_SESSIONS.update(unique)
    if EXCLUDE_TRAIN_OVERLAP:
        unique = {k: v for k, v in unique.items() if k not in overlap}
    report = {
        "validation_trials": len(rows),
        "distinct_sentences": len(sessions_of),
        "sentences_in_several_sessions": sum(len(v) > 1 for v in sessions_of.values()),
        "trials_whose_sentence_repeats": sum(len(sessions_of[r["key"]]) > 1 for r in rows),
        "cached_lists": len(cache),
        "cached_lists_with_unique_session": len(unique),
        "cached_lists_without_session": sum(k not in sessions_of for k in cache),
        "in_decoder_training_text": len(overlap),
        "training_text_overlap_excluded": EXCLUDE_TRAIN_OVERLAP,
    }
    return unique, report


def fit_temperature(cache, rec, keys, alpha):
    """Cinà's revision_analysis.fit_temperature at this project's acoustic scale.

    His version calls analyze.picks with the module default acoustic_scale (0.35), while the published
    decoder setting and every other computation here use 0.3.
    """
    picks = A.picks(cache, rec["scores"], alpha, ACOUSTIC_SCALE)
    best, best_ece = 1.0, 9e9
    for T in [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]:
        conf = np.array([softmax_conf(rec["scores"][s], T)[0] for s in keys])
        ok = np.array([picks[s] == s for s in keys])
        e = A.ece(conf, ok)["ece"]
        if e < best_ece:
            best, best_ece = T, e
    return best, best_ece


def wer(err, picks, keys):
    words = sum(len(s.split()) for s in keys)
    return 100 * sum(err(s, picks[s]) for s in keys) / words if words else float("nan")


def tune_alpha(err, cache, scores, keys):
    return min(ALPHAS, key=lambda a: wer(err, A.picks(cache, scores, a, ACOUSTIC_SCALE), keys))


def per_session(err, cache, scores, alphas, session_of):
    first = A.picks(cache, None, 0.0, ACOUSTIC_SCALE)
    rescored = {arm: A.picks(cache, scores[arm], alphas[arm], ACOUSTIC_SCALE) for arm in ARMS}
    by_session = defaultdict(list)
    for s, sess in session_of.items():
        by_session[sess].append(s)
    table = []
    for sess in sorted(by_session):
        keys = by_session[sess]
        row = {"session": sess, "sentences": len(keys), "words": sum(len(s.split()) for s in keys),
               "first_pass_wer": wer(err, first, keys)}
        for arm in ARMS:
            row[f"{arm}_wer"] = wer(err, rescored[arm], keys)
            row[f"{arm}_gain"] = row["first_pass_wer"] - row[f"{arm}_wer"]
            deltas = [err(s, rescored[arm][s]) - err(s, first[s]) for s in keys]
            row[f"{arm}_helped"] = sum(d < 0 for d in deltas)
            row[f"{arm}_harmed"] = sum(d > 0 for d in deltas)
        table.append(row)
    return table


def predictions_1_to_3(table):
    big = [r for r in table if r["sentences"] >= MIN_SESSION_SENTENCES]
    w = np.array([r["first_pass_wer"] for r in big])
    q1, q3 = np.percentile(w, [25, 75])
    easy, hard = w[w <= q1].mean(), w[w >= q3].mean()
    order = np.arange(len(big))
    out = {"sessions_with_min_sentences": len(big),
           "P1_hard_over_easy_quartile_wer": hard / easy if easy else float("inf"),
           "P1_easy_quartile_mean_wer": easy, "P1_hard_quartile_mean_wer": hard}
    for arm in ARMS:
        rho, p = spearmanr(w, [r[f"{arm}_gain"] for r in big])
        out[f"P2_spearman_firstpass_vs_{arm}_gain"] = {"rho": rho, "p": p}
    rho, p = spearmanr(order, w)
    out["P3_spearman_date_vs_firstpass"] = {"rho": rho, "p": p}
    return out


def temporal_split(session_of):
    boundary_map = SPLIT_SESSIONS or session_of
    counts = defaultdict(int)
    for sess in boundary_map.values():
        counts[sess] += 1
    ordered = sorted(counts)  # session names are tNN.YYYY.MM.DD, so lexical order is date order
    total, early, running = len(boundary_map), [], 0
    for sess in ordered:
        if running >= total * (1 - LATE_FRACTION):
            break
        early.append(sess)
        running += counts[sess]
    late = [sess for sess in ordered if sess not in early]
    by_session = defaultdict(list)
    for s, sess in session_of.items():
        by_session[sess].append(s)
    return ([s for sess in early for s in by_session[sess]],
            [s for sess in late for s in by_session[sess]], early, late)


def calibration(cache, scores, alpha, tune_keys, eval_keys):
    rec = {"scores": scores}
    T, tune_ece = fit_temperature(cache, rec, tune_keys, alpha)
    picks = A.picks(cache, scores, alpha, ACOUSTIC_SCALE)
    conf = np.array([softmax_conf(scores[s], T)[0] for s in eval_keys])
    ok = np.array([picks[s] == s for s in eval_keys])
    return {"temperature": T, "tune_ece": tune_ece, "eval_ece": A.ece(conf, ok)["ece"],
            "eval_sentence_accuracy": float(ok.mean())}


def main():
    participant_arg()
    err = Errors()
    cache, scores = load()
    session_of, mapping = session_map(cache)
    print(json.dumps(mapping, indent=1))

    dev, test = split(cache)
    alphas_random = {arm: tune_alpha(err, cache, scores[arm], dev) for arm in ARMS}
    table = per_session(err, cache, scores, alphas_random, session_of)
    p13 = predictions_1_to_3(table)

    early, late, early_sessions, late_sessions = temporal_split(session_of)
    dev_set = set(dev)
    late_clean = [s for s in late if s not in dev_set]  # late sentences the random dev split never saw
    p4, p5 = {}, {}
    for arm in ARMS:
        a_early = tune_alpha(err, cache, scores[arm], early)
        picks = lambda a: A.picks(cache, scores[arm], a, ACOUSTIC_SCALE)  # noqa: E731
        p4[arm] = {"alpha_random_dev": alphas_random[arm], "alpha_early_sessions": a_early,
                   "late_wer_first_pass": wer(err, A.picks(cache, None, 0, ACOUSTIC_SCALE), late_clean),
                   "late_wer_tuned_random": wer(err, picks(alphas_random[arm]), late_clean),
                   "late_wer_tuned_early": wer(err, picks(a_early), late_clean)}
        p4[arm]["difference"] = p4[arm]["late_wer_tuned_early"] - p4[arm]["late_wer_tuned_random"]
        random_test_clean = [s for s in test if s in session_of]
        p5[arm] = {"random_split": calibration(cache, scores[arm], alphas_random[arm], dev, random_test_clean),
                   "temporal_split": calibration(cache, scores[arm], a_early, early, late)}

    out = {"mapping": mapping, "alphas_random_dev": alphas_random, "per_session": table, "P1_P3": p13,
           "temporal": {"early_sessions": early_sessions, "late_sessions": late_sessions,
                        "early_sentences": len(early), "late_sentences": len(late),
                        "late_sentences_outside_random_dev": len(late_clean)},
           "P4": p4, "P5": p5}
    result_path("step1_sessions").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ("alphas_random_dev", "P1_P3", "temporal", "P4", "P5")},
                     indent=1, default=float))


if __name__ == "__main__":
    main()
