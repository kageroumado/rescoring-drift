"""Step 4, input: phoneme logits per validation trial, in the format Cinà's build_lists_kaldi.py reads.

Same model loading, input-layer rule and forward pass as step2_drift.py (a session the model was not trained
on uses the latest trained session's input layer), so the greedy PER recomputed from these logits must equal
step 2's per-session PER; the script checks that against results/step2_per_<name>.json and refuses to write
otherwise. Logits are stored as float16, as in Cinà's extract_logits.py, and pickled through the
`numpy.ndarray(shape, dtype, buffer)` constructor: the decoder's environment pins NumPy 1.24, which cannot
unpickle NumPy 2's default array pickles (they name `numpy._core`).

    python src/step4_logits.py --checkpoint <run>/checkpoint --name early --out <ssd>/step4/logits_early.pkl
    python src/step4_logits.py --checkpoint <released>/checkpoint --name released --after 2023.11.26 --out ...
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import editdistance
import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step2_drift as D  # noqa: E402


class PortablePickler(pickle.Pickler):
    """Pickles arrays as numpy.ndarray(shape, dtype, buffer), which NumPy 1 and 2 both load."""

    def reducer_override(self, obj):
        if isinstance(obj, np.ndarray):
            a = np.ascontiguousarray(obj)
            return np.ndarray, (a.shape, a.dtype.str, a.tobytes())
        return NotImplemented


def dump(obj, path: Path) -> None:
    with open(path, "wb") as f:
        PortablePickler(f, protocol=4).dump(obj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--name", required=True, help="the step 2 result to check against: step2_per_<name>.json")
    ap.add_argument("--after", help="only sessions later than this date (YYYY.MM.DD)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    device = torch.device(a.device)
    model, args = D.load(a.checkpoint, device)
    trained = args["dataset"]["sessions"]
    last_trained = max(trained, key=D.session_date)
    reference = {r["session"]: r for r in json.loads((D.RESULTS / f"step2_per_{a.name}.json").read_text())["sessions"]}

    trials = []
    for path in sorted(D.DATA.glob("t15.*/data_val.hdf5")):
        session = path.parent.name
        if a.after and session.split(".", 1)[1] <= a.after:
            continue
        layer = trained.index(session if session in trained else last_trained)
        edits = length = 0
        with h5py.File(path, "r") as f:
            for key in f.keys():
                g = f[key]
                x = torch.tensor(g["input_features"][:], dtype=torch.float32, device=device).unsqueeze(0)
                logits = D.runSingleDecodingStep(x, layer, model, args, device)[0]
                truth = [int(p) for p in g["seq_class_ids"][: int(g.attrs["seq_len"])]]
                edits += editdistance.eval(truth, D.greedy(logits))
                length += len(truth)
                label = g.attrs["sentence_label"]
                trials.append({"session": session, "trial": int(g.attrs["trial_num"]),
                               "block": int(g.attrs["block_num"]),
                               "sentence": label.decode() if isinstance(label, bytes) else str(label),
                               "logits": np.asarray(logits).astype(np.float16)})
        ref = reference[session]
        if (edits, length) != (ref["edits"], ref["phonemes"]):
            sys.exit(f"{session}: {edits}/{length} edits/phonemes, step 2 has {ref['edits']}/{ref['phonemes']}")
        print(f"{session}  PER {100 * edits / length:5.1f}%  matches step 2", flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    dump({"trials": trials, "checkpoint": str(a.checkpoint), "last_trained": last_trained}, a.out)
    print(f"{len(trials)} trials -> {a.out}")


if __name__ == "__main__":
    main()
