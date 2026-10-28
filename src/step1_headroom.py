"""Headroom for a correction gate: word error if each rescorer's change were kept only where it helps.

The oracle uses the reference, so it is an upper bound on any real gate, never a result.

    python src/step1_headroom.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step1_sessions as S  # noqa: E402

A = S.A


def main():
    err = S.Errors()
    cache, scores = S.load()
    session_of, _ = S.session_map(cache)
    dev, test = S.split(cache)
    keys = [s for s in test if s in session_of]
    words = sum(len(s.split()) for s in keys)
    first = A.picks(cache, None, 0.0, S.ACOUSTIC_SCALE)
    out = {"test_sentences": len(keys), "test_words": words,
           "first_pass_wer": 100 * sum(err(s, first[s]) for s in keys) / words}
    for arm in S.ARMS:
        alpha = S.tune_alpha(err, cache, scores[arm], dev)
        picks = A.picks(cache, scores[arm], alpha, S.ACOUSTIC_SCALE)
        e_first = {s: err(s, first[s]) for s in keys}
        e_arm = {s: err(s, picks[s]) for s in keys}
        changed = [s for s in keys if picks[s] != first[s]]
        out[arm] = {
            "alpha": alpha,
            "wer": 100 * sum(e_arm.values()) / words,
            "oracle_gate_wer": 100 * sum(min(e_first[s], e_arm[s]) for s in keys) / words,
            "sentences_changed": len(changed),
            "changes_helped": sum(e_arm[s] < e_first[s] for s in changed),
            "changes_harmed": sum(e_arm[s] > e_first[s] for s in changed),
            "changes_neutral": sum(e_arm[s] == e_first[s] for s in changed),
            "words_fixed": sum(max(0, e_first[s] - e_arm[s]) for s in keys),
            "words_broken": sum(max(0, e_arm[s] - e_first[s]) for s in keys),
        }
    (S.RESULTS / "step1_headroom.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
