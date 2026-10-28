"""Map every evaluation sentence to the session, block and trial it was recorded in.

Reads only trial attributes and transcripts (never the neural features) and writes one row per trial, with
the sentence normalized the way Cinà's build_lists_kaldi.py normalizes the keys of his candidate lists.
T15: each session's data_val.hdf5 -> results/val_sessions.json. T12: the `test` split of the Dryad
competitionData .mat files, the trials Cinà's T12 lists were decoded from -> results/t12_sessions.json.

    python src/sessions.py [--participant t12]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import T12_DATA, T15_DATA  # noqa: E402
RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = {"t15": RESULTS / "val_sessions.json", "t12": RESULTS / "t12_sessions.json"}
PUNCTUATION = str.maketrans("", "", ",.?!-’'\";:()[]")


def normalize(text: str) -> str:
    return " ".join(text.translate(PUNCTUATION).lower().split())


def t15_rows() -> list[dict]:
    import h5py

    rows = []
    for path in sorted(T15_DATA.glob("t15.*/data_val.hdf5")):
        with h5py.File(path, "r") as f:
            for key in f.keys():
                attrs = f[key].attrs
                label = attrs["sentence_label"]
                label = label.decode() if isinstance(label, bytes) else str(label)
                rows.append({
                    "session": str(attrs["session"]),
                    "block": int(attrs["block_num"]),
                    "trial": int(attrs["trial_num"]),
                    "sentence": label,
                    "key": normalize(label),
                })
    return rows


def t12_rows() -> list[dict]:
    import scipy.io

    rows = []
    for path in sorted(T12_DATA.glob("t12.*.mat")):
        m = scipy.io.loadmat(path, variable_names=["sentenceText", "blockIdx"])
        for trial, (text, block) in enumerate(zip(m["sentenceText"], m["blockIdx"].ravel())):
            label = str(text).strip()
            rows.append({"session": path.stem, "block": int(block), "trial": trial,
                         "sentence": label, "key": normalize(label)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", choices=["t15", "t12"], default="t15")
    participant = ap.parse_args().participant
    rows = t12_rows() if participant == "t12" else t15_rows()
    out = OUT[participant]
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    sessions = sorted({r["session"] for r in rows})
    print(f"{len(rows)} trials across {len(sessions)} sessions -> {out}")


if __name__ == "__main__":
    main()
