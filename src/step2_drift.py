"""Step 2: phoneme error rate per session for a decoder trained on early sessions only (predictions D1-D3).

Runs a checkpoint greedily (no language model) over every session's validation trials and scores each
against its phoneme labels. A session the model has no input layer for uses the input layer of the latest
session it was trained on: the decoder as deployed with no recalibration. Decoding follows standard CTC
(merge repeats, then drop blanks); the official evaluate_model.py drops blanks first, which merges a
genuinely doubled phoneme, so its PER can differ slightly.

    python src/step2_drift.py --checkpoint <run>/checkpoint --name early
    python src/step2_drift.py --checkpoint <dryad>/t15_pretrained_rnn_baseline/checkpoint --name released
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import editdistance
import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import NEJM_TRAINING as NEJM, T15_DATA as DATA  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
sys.path.insert(0, str(NEJM))
from rnn_model import GRUDecoder  # noqa: E402
from evaluate_model_helpers import runSingleDecodingStep  # noqa: E402


def session_date(session: str) -> date:
    y, m, d = (int(x) for x in session.split(".")[1:])
    return date(y, m, d)


def load(checkpoint: Path, device):
    args = OmegaConf.to_container(OmegaConf.load(checkpoint / "args.yaml"), resolve=True)
    args["use_amp"] = False
    m = args["model"]
    model = GRUDecoder(neural_dim=m["n_input_features"], n_units=m["n_units"],
                       n_days=len(args["dataset"]["sessions"]), n_classes=args["dataset"]["n_classes"],
                       rnn_dropout=m["rnn_dropout"], input_dropout=m["input_network"]["input_layer_dropout"],
                       n_layers=m["n_layers"], patch_size=m["patch_size"], patch_stride=m["patch_stride"])
    # The checkpoint's pickle imports only tensors, an OrderedDict and NumPy's scalar and dtype types
    # (the saved validation metrics); the weights-only loader is widened by exactly those.
    # NumPy 1 pickles the scalar type as numpy.core.multiarray, NumPy 2 (the CUDA pod) as numpy._core.
    allowed = [(np._core.multiarray.scalar, "numpy.core.multiarray.scalar"),
               (np._core.multiarray.scalar, "numpy._core.multiarray.scalar"), np.dtype,
               np.dtypes.Float64DType, np.dtypes.Float32DType, np.dtypes.Int64DType]
    with torch.serialization.safe_globals(allowed):
        state = torch.load(checkpoint / "best_checkpoint", map_location="cpu", weights_only=True)
    weights = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in state["model_state_dict"].items()}
    model.load_state_dict(weights)
    return model.to(device).eval(), args


def greedy(logits: np.ndarray) -> list[int]:
    best = logits.argmax(-1)
    merged = [int(p) for i, p in enumerate(best) if i == 0 or p != best[i - 1]]
    return [p for p in merged if p != 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    device = torch.device(a.device)
    model, args = load(a.checkpoint, device)
    trained = args["dataset"]["sessions"]
    last_trained = max(trained, key=session_date)

    rows = []
    for path in sorted(DATA.glob("t15.*/data_val.hdf5")):
        session = path.parent.name
        layer_session = session if session in trained else last_trained
        layer = trained.index(layer_session)
        edits = length = 0
        with h5py.File(path, "r") as f:
            for key in f.keys():
                g = f[key]
                x = torch.tensor(g["input_features"][:], dtype=torch.float32, device=device).unsqueeze(0)
                truth = [int(p) for p in g["seq_class_ids"][: int(g.attrs["seq_len"])]]
                pred = greedy(runSingleDecodingStep(x, layer, model, args, device)[0])
                edits += editdistance.eval(truth, pred)
                length += len(truth)
        rows.append({"session": session, "trained_on": session in trained, "input_layer": layer_session,
                     "days_after_training": max(0, (session_date(session) - session_date(last_trained)).days),
                     "phonemes": length, "edits": edits, "per": edits / length})
        print(f"{session}  {'train' if session in trained else 'later'}  PER {100 * edits / length:5.1f}%", flush=True)

    out = {"checkpoint": str(a.checkpoint), "trained_sessions": len(trained), "last_trained": last_trained,
           "sessions": rows}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"step2_per_{a.name}.json").write_text(json.dumps(out, indent=1))
    print(f"-> results/step2_per_{a.name}.json")


if __name__ == "__main__":
    main()
