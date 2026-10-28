"""Step 5a: does the input layer carry the drift? The released checkpoint, whose shared network was
trained on all 45 sessions, is run on every late session through the 2023.11.26 input layer instead
of the session's own. Holding the shared weights fixed and varying only the input adaptation separates
what the early model (step 2) lost through its input layer from what it lost through a shared network
that never saw the later sessions.

    python src/step5_input_layer.py        # writes results/step5_input_layer.json
"""
import argparse
import json
import sys
from pathlib import Path

import editdistance
import h5py
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step2_drift as D  # noqa: E402
from paths import RELEASED_CHECKPOINT  # noqa: E402

FROZEN = "t15.2023.11.26"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=RELEASED_CHECKPOINT)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    device = torch.device(a.device)
    model, args = D.load(a.checkpoint, device)
    trained = args["dataset"]["sessions"]
    frozen_layer = trained.index(FROZEN)
    own = {r["session"]: r for r in json.loads((D.RESULTS / "step2_per_released.json").read_text())["sessions"]}
    early = {r["session"]: r for r in json.loads((D.RESULTS / "step2_per_early.json").read_text())["sessions"]}
    rows = []
    for path in sorted(D.DATA.glob("t15.*/data_val.hdf5")):
        session = path.parent.name
        if D.session_date(session) <= D.session_date(FROZEN):
            continue
        edits = length = 0
        with h5py.File(path, "r") as f:
            for key in f.keys():
                g = f[key]
                x = torch.tensor(g["input_features"][:], dtype=torch.float32, device=device).unsqueeze(0)
                truth = [int(p) for p in g["seq_class_ids"][: int(g.attrs["seq_len"])]]
                pred = D.greedy(D.runSingleDecodingStep(x, frozen_layer, model, args, device)[0])
                edits += editdistance.eval(truth, pred)
                length += len(truth)
        rows.append({"session": session, "days_after_training": early[session]["days_after_training"],
                     "phonemes": length, "edits_frozen_layer": edits, "per_frozen_layer": edits / length,
                     "per_own_layer": own[session]["per"], "per_early_model": early[session]["per"]})
        print(f"{session}  frozen {100 * edits / length:5.1f}%  own {100 * own[session]['per']:5.1f}%  "
              f"early model {100 * early[session]['per']:5.1f}%", flush=True)

    def weighted(key):
        return sum(r[key] * r["phonemes"] for r in rows) / sum(r["phonemes"] for r in rows)

    out = {"checkpoint": str(a.checkpoint), "frozen_input_layer": FROZEN, "sessions": rows,
           "late_mean_per": {"released_own_layer": weighted("per_own_layer"), "released_frozen_layer": weighted("per_frozen_layer"),
                             "early_model": weighted("per_early_model")}}
    (D.RESULTS / "step5_input_layer.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["late_mean_per"], indent=1))


if __name__ == "__main__":
    main()
