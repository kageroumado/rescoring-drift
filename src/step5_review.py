"""Step 5b: the CPU-side review analyses (predictions S1, S3, S4 in PREDICTIONS.md).

S1  Near-regime stability: per-session paired gate-minus-always differences and leave-one-session-out
    verdicts, with the gate fit once on the drifted decoder's training sessions.
S3  Random-feature control for the probe: the step 3b pipeline with Gaussian random features in place of
    the activation differences, same selection and threshold tuning, 20 seeds.
S4  Feature distributions: standardized means of the eight gate features for accepted/vetoed and
    helpful/harmful changes on the calibrated and the drifted decoder, and list length on matched
    sessions.

    python src/step5_review.py          # writes results/step5_review.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step1_sessions as S  # noqa: E402
import step3a_gate as G  # noqa: E402
import step3b_probe as P  # noqa: E402
import step4_rescoring as R4  # noqa: E402

SEEDS = 20
RANDOM_DIM = 64            # per-hypothesis random feature vector, reduced by PCA like the activations
RANDOM_LAYERS = 8          # the probe chose among 8 layers; the control chooses among 8 random draws


def fit_gate(setting, arm):
    """The step 3a gate for one arm: per-sentence errors on the test set under gate and always-apply,
    and the fitted model, scaler and threshold."""
    train_changed = setting.changed(arm, setting.train)
    X = np.array([setting.features(arm, s) for s in train_changed])
    y = np.array([setting.err(s, setting.picks[arm][s]) < setting.err(s, setting.first[s]) for s in train_changed])
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(C=1.0, max_iter=1000).fit(scaler.transform(X), y)

    def prob(keys):
        return dict(zip(keys, model.predict_proba(scaler.transform(np.array([setting.features(arm, s) for s in keys])))[:, 1])) if keys else {}

    threshold = G.tune_threshold(setting, arm, prob(train_changed))
    p_test = prob(setting.changed(arm, setting.test))
    accept = lambda s: s in p_test and p_test[s] >= threshold  # noqa: E731
    return {"model": model, "scaler": scaler, "threshold": threshold, "p_test": p_test, "accept": accept,
            "e_gate": setting.errors(arm, setting.test, accept),
            "e_always": setting.errors(arm, setting.test, lambda s: True)}


def s1_near_stability(err):
    S.configure("t15")
    S.ARMS = ["opt", "qwen"]
    cache, _, scores = R4.load_lists(R4.RESULTS_DIR, "early")
    session_of, _ = S.session_map(cache)
    keys = R4.keys_by_regime(session_of, R4.regimes())
    setting = G.Setting(err, cache, scores, keys["train"], keys["near"])
    words = np.array([len(s.split()) for s in setting.test], float)
    sessions = np.array([session_of[s] for s in setting.test])
    out = {}
    for arm in S.ARMS:
        g = fit_gate(setting, arm)
        diff = g["e_gate"] - g["e_always"]
        full = float(diff.sum() / words.sum() * 100)
        per_session, loso = {}, {}
        for sess in sorted(set(sessions)):
            m = sessions == sess
            per_session[sess] = {"sentences": int(m.sum()), "words": int(words[m].sum()),
                                 "gate_minus_always_points": float(diff[m].sum() / words[m].sum() * 100),
                                 "gate_minus_always_words": int(diff[m].sum())}
            loso[sess] = float(diff[~m].sum() / words[~m].sum() * 100)
        out[arm] = {"full_gate_minus_always": full, "per_session": per_session, "leave_one_out": loso,
                    "sign_changes_under_loso": sum((v > 0) != (full > 0) for v in loso.values()) if full != 0 else None}
    return out


def s3_random_features(err):
    S.configure("t15")
    S.ARMS = ["opt", "qwen", "jev"]
    cache, scores = S.load()
    session_of, _ = S.session_map(cache)
    early, late, _, _ = S.temporal_split(session_of)
    setting = G.Setting(err, cache, scores, early, late)
    e_gate, _ = P.gate_3a_errors(setting)
    words = np.array([len(s.split()) for s in setting.test], float)
    hyps = set()
    for s in setting.changed(P.ARM, setting.train) + setting.changed(P.ARM, setting.test):
        hyps.update([setting.first[s], setting.picks[P.ARM][s]])
    hyps = sorted(hyps)
    deltas, aucs, applied = [], [], []
    for seed in range(SEEDS):
        rng = np.random.default_rng(seed)
        acts = {h: {"last": rng.normal(size=(RANDOM_LAYERS, RANDOM_DIM)).astype(np.float32)} for h in hyps}
        r = P.evaluate(setting, session_of, acts, list(range(RANDOM_LAYERS)), "last")
        deltas.append(float((r["errors"].sum() - e_gate.sum()) / words.sum() * 100))
        aucs.append(r["test_auc_helped"])
        applied.append(r["applied"])
    real = json.loads((S.RESULTS / "step3b_probe.json").read_text())["temporal"]["probe_last"]
    d = np.array(deltas)
    return {"seeds": SEEDS, "random_dim": RANDOM_DIM, "random_layers": RANDOM_LAYERS,
            "minus_gate_3a": {"values": deltas, "median": float(np.median(d)), "mean": float(d.mean()),
                              "p5": float(np.percentile(d, 5)), "p95": float(np.percentile(d, 95)),
                              "share_worse_than_gate": float((d > 0).mean())},
            "test_auc_helped": {"values": aucs, "median": float(np.median([a for a in aucs if a is not None]))},
            "real_probe_minus_gate_3a": real["minus_gate_3a"],
            "real_inside_central_90": bool(np.percentile(d, 5) <= real["minus_gate_3a"] <= np.percentile(d, 95)),
            "share_of_seeds_worse_than_real": float((d > real["minus_gate_3a"]).mean())}


def feature_table(setting, arm, g):
    """Standardized feature means (fit-set scaler) by accepted/vetoed and helpful/harmful on the test set."""
    changed = setting.changed(arm, setting.test)
    X = g["scaler"].transform(np.array([setting.features(arm, s) for s in changed]))
    helped = np.array([setting.err(s, setting.picks[arm][s]) < setting.err(s, setting.first[s]) for s in changed])
    harmed = np.array([setting.err(s, setting.picks[arm][s]) > setting.err(s, setting.first[s]) for s in changed])
    accepted = np.array([g["accept"](s) for s in changed])
    groups = {"accepted": accepted, "vetoed": ~accepted, "helpful": helped, "harmful": harmed}
    table = {}
    for i, name in enumerate(G.FEATURES):
        row = {k: float(X[m, i].mean()) if m.any() else None for k, m in groups.items()}
        row["helpful_minus_harmful"] = (row["helpful"] - row["harmful"]) if row["helpful"] is not None and row["harmful"] is not None else None
        row["coefficient"] = float(g["model"].coef_[0][i])
        table[name] = row
    return {"changes": len(changed), "accepted": int(accepted.sum()), "helpful": int(helped.sum()),
            "harmful": int(harmed.sum()), "features": table}


def s4_feature_distributions(err):
    out = {}
    S.configure("t15")
    S.ARMS = ["opt", "qwen", "jev"]
    cache, scores = S.load()
    session_of, _ = S.session_map(cache)
    early, late, _, _ = S.temporal_split(session_of)
    setting = G.Setting(err, cache, scores, early, late)
    out["t15_calibrated"] = {arm: feature_table(setting, arm, fit_gate(setting, arm)) for arm in ("opt", "qwen")}
    S.ARMS = ["opt", "qwen"]
    early_cache, _, early_scores = R4.load_lists(R4.RESULTS_DIR, "early")
    rel_cache, _, _ = R4.load_lists(R4.RESULTS_DIR, "released_late")
    e_map, _ = S.session_map(early_cache)
    r_map, _ = S.session_map(rel_cache)
    keys = R4.keys_by_regime(e_map, R4.regimes())
    near = G.Setting(err, early_cache, early_scores, keys["train"], keys["near"])
    out["t15_drift_near"] = {arm: feature_table(near, arm, fit_gate(near, arm)) for arm in S.ARMS}
    late_sessions = {s for s in r_map.values()}
    early_on_late = [len(early_cache[k]) for k, sess in e_map.items() if sess in late_sessions]
    released_late = [len(rel_cache[k]) for k in r_map]
    out["list_length_matched_late_sessions"] = {
        "early_model": {"lists": len(early_on_late), "mean": float(np.mean(early_on_late))},
        "released_model": {"lists": len(released_late), "mean": float(np.mean(released_late))},
        "early_model_all_sessions_mean": float(np.mean([len(v) for v in early_cache.values()]))}
    return out


def main():
    err = S.Errors()
    out = {"S1_near_stability": s1_near_stability(err),
           "S4_feature_distributions": s4_feature_distributions(err),
           "S3_random_features": s3_random_features(err)}
    (S.RESULTS / "step5_review.json").write_text(json.dumps(out, indent=1, default=float))
    s1 = out["S1_near_stability"]
    for arm, r in s1.items():
        print(f"S1 {arm}: full {r['full_gate_minus_always']:+.2f}; per session",
              {k: round(v["gate_minus_always_points"], 2) for k, v in r["per_session"].items()},
              "; LOSO", {k: round(v, 2) for k, v in r["leave_one_out"].items()})
    s3 = out["S3_random_features"]["minus_gate_3a"]
    print(f"S3 random features minus gate: median {s3['median']:+.2f}, 5–95% {s3['p5']:+.2f}..{s3['p95']:+.2f}, "
          f"share worse {s3['share_worse_than_gate']:.2f}; real probe {out['S3_random_features']['real_probe_minus_gate_3a']:+.2f}")
    for name, blocks in out["S4_feature_distributions"].items():
        if name.startswith("t15"):
            for arm, t in blocks.items():
                print(f"S4 {name} {arm}: agreement helpful−harmful {t['features']['rescorer_agreement']['helpful_minus_harmful']:+.2f} sd")
    print("S4 list length on late sessions:", out["S4_feature_distributions"]["list_length_matched_late_sessions"])


if __name__ == "__main__":
    main()
