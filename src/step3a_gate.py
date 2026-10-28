"""Step 3a: a correct-or-keep gate from output features (predictions Q1-Q4 in PREDICTIONS.md).

For every sentence where a rescorer's pick differs from the first pass, the gate decides whether to apply
the change. Everything fitted (rescoring weight α, softmax temperature, logistic weights, thresholds) is
fitted on the training split only. Primary split: the earliest sessions holding half the sentences -> the
rest (T15: 21 -> 20). Secondary: Cinà's random 30/70 split.

    python src/step3a_gate.py [--participant t12]
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step1_sessions as S  # noqa: E402

A = S.A
FEATURES = ["first_pass_margin", "rescorer_margin", "pick_rank", "edit_distance", "length_words",
            "list_length", "rescorer_agreement", "softmax_confidence"]
THRESHOLDS = np.round(np.arange(0.0, 1.001, 0.02), 2)
BOOTSTRAP = 5000


def z(v):
    v = np.asarray(v, dtype=float)
    return (v - v.mean()) / (v.std() + 1e-9)


class Setting:
    """One split: everything fitted on `train`, everything reported on `test`."""

    def __init__(self, err, cache, scores, train, test):
        self.err, self.cache, self.scores, self.train, self.test = err, cache, scores, train, test
        self.first = A.picks(cache, None, 0.0, S.ACOUSTIC_SCALE)
        self.alpha = {arm: S.tune_alpha(err, cache, scores[arm], train) for arm in S.ARMS}
        self.picks = {arm: A.picks(cache, scores[arm], self.alpha[arm], S.ACOUSTIC_SCALE) for arm in S.ARMS}
        self.temperature = {arm: S.fit_temperature(cache, {"scores": scores[arm]}, train, self.alpha[arm])[0]
                            for arm in S.ARMS}

    def features(self, arm, s):
        nb = self.cache[s]
        hyps = [h for h, _, _ in nb]
        first_total = np.array([S.ACOUSTIC_SCALE * a + l for _, a, l in nb])
        i0, ia = hyps.index(self.first[s]), hyps.index(self.picks[arm][s])
        zf, zr = z(first_total), z(self.scores[arm][s])
        order = np.argsort(-first_total)
        agreement = sum(self.picks[other][s] == self.picks[arm][s] for other in S.ARMS)
        return [zf[i0] - zf[ia], zr[ia] - zr[i0], int(np.where(order == ia)[0][0]),
                self.err(self.first[s], self.picks[arm][s]), len(self.first[s].split()), len(nb),
                agreement, S.softmax_conf(self.scores[arm][s], self.temperature[arm])[0]]

    def changed(self, arm, keys):
        return [s for s in keys if self.picks[arm][s] != self.first[s]]

    def errors(self, arm, keys, apply):
        """Word errors per sentence when the change is applied where apply(s) is true."""
        return np.array([self.err(s, self.picks[arm][s] if apply(s) else self.first[s]) for s in keys], float)


def tune_threshold(setting, arm, values):
    keys = setting.train
    return min(THRESHOLDS, key=lambda t: setting.errors(
        arm, keys, lambda s: s in values and values[s] >= t).sum())


def cluster_bootstrap(diff, groups):
    """95% interval of the summed difference / words, resampling whole sessions."""
    ids = sorted(set(groups))
    rng = np.random.default_rng(0)
    idx = {g: np.where(np.array(groups) == g)[0] for g in ids}
    draws = []
    for _ in range(BOOTSTRAP):
        pick = rng.choice(ids, len(ids))
        rows = np.concatenate([idx[g] for g in pick])
        draws.append(diff[rows, 0].sum() / diff[rows, 1].sum() * 100)
    return [float(x) for x in np.percentile(draws, [2.5, 97.5])]


def run(setting, session_of):
    out = {"alpha": setting.alpha, "temperature": setting.temperature, "arms": {}}
    test = setting.test
    words = np.array([len(s.split()) for s in test], float)
    groups = [session_of[s] for s in test]
    for arm in S.ARMS:
        train_changed = setting.changed(arm, setting.train)
        X = np.array([setting.features(arm, s) for s in train_changed])
        y = np.array([setting.err(s, setting.picks[arm][s]) < setting.err(s, setting.first[s])
                      for s in train_changed])
        if len(set(y)) < 2:
            # α tuned to 0 changes nothing, and one-sided labels leave nothing to separate: no gate to fit.
            out["arms"][arm] = {"train_changed": len(train_changed), "train_helped": int(y.sum()),
                                "test_changed": len(setting.changed(arm, test)), "gate": "not fittable"}
            continue
        scaler = StandardScaler().fit(X)
        model = LogisticRegression(C=1.0, max_iter=1000).fit(scaler.transform(X), y)

        def probabilities(keys):
            ch = setting.changed(arm, keys)
            if not ch:
                return {}
            p = model.predict_proba(scaler.transform(np.array([setting.features(arm, s) for s in ch])))[:, 1]
            return dict(zip(ch, p))

        def confidences(keys):
            return {s: setting.features(arm, s)[-1] for s in setting.changed(arm, keys)}

        t_gate = tune_threshold(setting, arm, probabilities(setting.train))
        t_conf = tune_threshold(setting, arm, confidences(setting.train))
        p_test, c_test = probabilities(test), confidences(test)
        policies = {
            "none": lambda s: False,
            "always": lambda s: True,
            "softmax_threshold": lambda s: s in c_test and c_test[s] >= t_conf,
            "logistic_gate": lambda s: s in p_test and p_test[s] >= t_gate,
        }
        e_first = setting.errors(arm, test, policies["none"])
        e_always = setting.errors(arm, test, policies["always"])
        result = {"train_changed": len(train_changed), "train_helped": int(y.sum()),
                  "test_changed": len(setting.changed(arm, test)),
                  "threshold_logistic": float(t_gate), "threshold_softmax": float(t_conf),
                  "standardized_coefficients": dict(zip(FEATURES, map(float, model.coef_[0]))),
                  "wer": {}, "applied": {}}
        for name, policy in policies.items():
            e = setting.errors(arm, test, policy)
            result["wer"][name] = float(e.sum() / words.sum() * 100)
            applied = [s for s in setting.changed(arm, test) if policy(s)]
            result["applied"][name] = {
                "changes": len(applied),
                "helped": sum(setting.err(s, setting.picks[arm][s]) < setting.err(s, setting.first[s]) for s in applied),
                "harmed": sum(setting.err(s, setting.picks[arm][s]) > setting.err(s, setting.first[s]) for s in applied),
            }
            if name not in ("none", "always"):
                diff = np.stack([e - e_always, words], axis=1)
                result.setdefault("vs_always_ci", {})[name] = cluster_bootstrap(diff, groups)
        result["wer"]["oracle"] = float(np.minimum(e_first, e_always).sum() / words.sum() * 100)
        headroom = result["wer"]["always"] - result["wer"]["oracle"]
        result["share_of_headroom_recovered"] = (
            (result["wer"]["always"] - result["wer"]["logistic_gate"]) / headroom if headroom else None)
        out["arms"][arm] = result
    return out


def main():
    S.participant_arg()
    err = S.Errors()
    cache, scores = S.load()
    session_of, _ = S.session_map(cache)
    early, late, _, _ = S.temporal_split(session_of)
    dev, test = S.split(cache)
    settings = {
        "temporal": Setting(err, cache, scores, early, late),
        "random": Setting(err, cache, scores, [s for s in dev if s in session_of],
                          [s for s in test if s in session_of]),
    }
    out = {name: run(setting, session_of) for name, setting in settings.items()}
    S.result_path("step3a_gate").write_text(json.dumps(out, indent=1, default=float))
    for name, res in out.items():
        print(f"\n=== {name} split ===")
        for arm, r in res["arms"].items():
            if "wer" not in r:
                print(f"{arm:5} changed train {r['train_changed']:3} (helped {r['train_helped']:3}) "
                      f"test {r['test_changed']:3} | gate not fittable")
                continue
            w = r["wer"]
            print(f"{arm:5} changed train {r['train_changed']:3} (helped {r['train_helped']:3}) test {r['test_changed']:3} | "
                  f"none {w['none']:.2f} always {w['always']:.2f} softmax {w['softmax_threshold']:.2f} "
                  f"gate {w['logistic_gate']:.2f} oracle {w['oracle']:.2f} | headroom recovered "
                  f"{r['share_of_headroom_recovered']:.0%}")
            print(f"      gate vs always 95% CI {r['vs_always_ci']['logistic_gate']}  softmax vs always "
                  f"{r['vs_always_ci']['softmax_threshold']}")
            print(f"      applied: " + "; ".join(f"{k} {v['changes']} (+{v['helped']}/-{v['harmed']})"
                                                 for k, v in r["applied"].items()))
            top = sorted(r["standardized_coefficients"].items(), key=lambda kv: -abs(kv[1]))[:3]
            print(f"      top coefficients: " + ", ".join(f"{k} {v:+.2f}" for k, v in top))


if __name__ == "__main__":
    main()
