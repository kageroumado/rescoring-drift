"""Step 3b, part 2: does a probe on OPT-6.7b activations improve the step 3a gate? (prediction Q5)

For each sentence where OPT's pick differs from the first pass, the probe sees the activation difference
h(OPT's pick) - h(first pass) at one layer, final token (the predicted setting; mean pooling is reported as
exploratory). The difference is reduced with PCA fit on the training split, then joined to the step 3a
features in the same logistic gate. The layer and the number of components are chosen by session-grouped
cross-validation on the training split; the probability threshold is tuned on training WER, as in 3a.

    python src/step3b_probe.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step1_sessions as S  # noqa: E402
import step3a_gate as G  # noqa: E402

from paths import ACTIVATIONS  # noqa: E402

COMPONENTS = [2, 4, 8, 16]
ARM = "opt"


class Probe:
    """PCA on activation differences + the 3a features, standardized, L2 logistic (C = 1)."""

    def __init__(self, layer_index, components, pooling):
        self.layer_index, self.components, self.pooling = layer_index, components, pooling

    def raw(self, setting, acts, keys):
        diffs = np.array([acts[setting.picks[ARM][s]][self.pooling][self.layer_index].astype(float)
                          - acts[setting.first[s]][self.pooling][self.layer_index].astype(float) for s in keys])
        feats = np.array([setting.features(ARM, s) for s in keys])
        return diffs, feats

    def fit(self, setting, acts, keys, y):
        diffs, feats = self.raw(setting, acts, keys)
        self.pca = PCA(n_components=min(self.components, len(keys) - 1), random_state=0).fit(diffs)
        X = np.hstack([self.pca.transform(diffs), feats])
        self.scaler = StandardScaler().fit(X)
        self.model = LogisticRegression(C=1.0, max_iter=2000).fit(self.scaler.transform(X), y)
        return self

    def predict(self, setting, acts, keys):
        diffs, feats = self.raw(setting, acts, keys)
        X = np.hstack([self.pca.transform(diffs), feats])
        return self.model.predict_proba(self.scaler.transform(X))[:, 1]


def labels(setting, keys):
    return np.array([setting.err(s, setting.picks[ARM][s]) < setting.err(s, setting.first[s]) for s in keys])


def choose(setting, acts, train_changed, y, groups, layers, pooling):
    """Layer and components by 5-fold session-grouped CV log loss on the training split."""
    best = None
    folds = list(GroupKFold(n_splits=5).split(train_changed, y, groups))
    for li in range(len(layers)):
        for k in COMPONENTS:
            losses = []
            for tr, va in folds:
                if len(set(y[tr])) < 2:
                    continue
                keys_tr = [train_changed[i] for i in tr]
                keys_va = [train_changed[i] for i in va]
                p = Probe(li, k, pooling).fit(setting, acts, keys_tr, y[tr]).predict(setting, acts, keys_va)
                losses.append(log_loss(y[va], np.clip(p, 1e-6, 1 - 1e-6), labels=[False, True]))
            score = float(np.mean(losses))
            if best is None or score < best[0]:
                best = (score, li, k)
    return best


def evaluate(setting, session_of, acts, layers, pooling):
    train_changed = setting.changed(ARM, setting.train)
    y = labels(setting, train_changed)
    groups = [session_of[s] for s in train_changed]
    cv_loss, li, k = choose(setting, acts, train_changed, y, groups, layers, pooling)
    probe = Probe(li, k, pooling).fit(setting, acts, train_changed, y)
    p_train = dict(zip(train_changed, probe.predict(setting, acts, train_changed)))
    threshold = G.tune_threshold(setting, ARM, p_train)

    test = setting.test
    test_changed = setting.changed(ARM, test)
    p_test = dict(zip(test_changed, probe.predict(setting, acts, test_changed)))
    y_test = labels(setting, test_changed)
    words = np.array([len(s.split()) for s in test], float)
    groups_test = [session_of[s] for s in test]
    e_probe = setting.errors(ARM, test, lambda s: s in p_test and p_test[s] >= threshold)
    applied = [s for s in test_changed if p_test[s] >= threshold]
    return {
        "pooling": pooling, "layer": layers[li], "components": k, "cv_log_loss": cv_loss,
        "threshold": float(threshold),
        "test_auc_helped": float(roc_auc_score(y_test, [p_test[s] for s in test_changed]))
        if len(set(y_test)) == 2 else None,
        "wer": float(e_probe.sum() / words.sum() * 100),
        "applied": {"changes": len(applied),
                    "helped": int(sum(labels(setting, applied))) if applied else 0,
                    "harmed": sum(setting.err(s, setting.picks[ARM][s]) > setting.err(s, setting.first[s])
                                  for s in applied)},
        "errors": e_probe, "words": words, "groups": groups_test,
    }


def gate_3a_errors(setting):
    """Reproduce the step 3a gate's per-sentence errors for OPT, for a paired comparison."""
    train_changed = setting.changed(ARM, setting.train)
    X = np.array([setting.features(ARM, s) for s in train_changed])
    y = labels(setting, train_changed)
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(C=1.0, max_iter=1000).fit(scaler.transform(X), y)
    prob = lambda keys: dict(zip(keys, model.predict_proba(  # noqa: E731
        scaler.transform(np.array([setting.features(ARM, s) for s in keys])))[:, 1])) if keys else {}
    threshold = G.tune_threshold(setting, ARM, prob(train_changed))
    test_changed = setting.changed(ARM, setting.test)
    p = prob(test_changed)
    y_test = labels(setting, test_changed)
    auc = (float(roc_auc_score(y_test, [p[s] for s in test_changed])) if len(set(y_test)) == 2 else None)
    return setting.errors(ARM, setting.test, lambda s: s in p and p[s] >= threshold), auc


def main():
    data = np.load(ACTIVATIONS)
    layers = [int(x) for x in data["layers"]]
    acts = {t: {"last": data["last"][i], "mean": data["mean"][i]} for i, t in enumerate(data["texts"])}
    err = S.Errors()
    cache, scores = S.load()
    session_of, _ = S.session_map(cache)
    early, late, _, _ = S.temporal_split(session_of)
    dev, test = S.split(cache)
    settings = {
        "temporal": G.Setting(err, cache, scores, early, late),
        "random": G.Setting(err, cache, scores, [s for s in dev if s in session_of],
                            [s for s in test if s in session_of]),
    }
    out = {}
    for name, setting in settings.items():
        e_gate, auc_gate = gate_3a_errors(setting)
        e_always = setting.errors(ARM, setting.test, lambda s: True)
        res = {"gate_3a": {"wer": None, "test_auc_helped": auc_gate}}
        for pooling in ("last", "mean"):
            r = evaluate(setting, session_of, acts, layers, pooling)
            words, groups = r.pop("words"), r.pop("groups")
            e = r.pop("errors")
            res["gate_3a"]["wer"] = float(e_gate.sum() / words.sum() * 100)
            res["always_wer"] = float(e_always.sum() / words.sum() * 100)
            r["minus_gate_3a"] = float((e.sum() - e_gate.sum()) / words.sum() * 100)
            r["minus_gate_3a_ci"] = G.cluster_bootstrap(np.stack([e - e_gate, words], axis=1), groups)
            res[f"probe_{pooling}" + ("" if pooling == "last" else "_exploratory")] = r
        out[name] = res
    (S.RESULTS / "step3b_probe.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
