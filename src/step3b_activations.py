"""Step 3b, part 1: OPT-6.7b residual-stream activations for every hypothesis the gate has to judge.

For each sentence where OPT's pick differs from the first pass (under the temporal or the random split's
α), runs both hypotheses through OPT-6.7b exactly as Cinà's ARScoreRescorer scores them (raw sentence,
the model's own tokenizer and BOS, length-normalized log-likelihood) and stores hidden states at every
4th layer: the final token and the mean over sentence tokens. Also rescores 30 full candidate lists and
compares with Cinà's cached OPT scores, to check that this setup reproduces his.

Output (73 MB, outside the repository): $RESCORING_DRIFT_DATA/activations/opt_step3b.npz

    HF_HOME=$RESCORING_DRIFT_DATA/hf python -I src/step3b_activations.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step1_sessions as S  # noqa: E402
from paths import ACTIVATIONS as OUT  # noqa: E402

MODEL = "facebook/opt-6.7b"
REVISION = "a45aa65bbeb77c1558bc99bedc6779195462dab0"
LAYERS = [4, 8, 12, 16, 20, 24, 28, 32]
CHECK_LISTS = 30


def hypotheses_to_run(err, cache, scores, session_of):
    early, late, _, _ = S.temporal_split(session_of)
    dev, test = S.split(cache)
    dev = [s for s in dev if s in session_of]
    first = S.A.picks(cache, None, 0.0, S.ACOUSTIC_SCALE)
    texts = set()
    for train in (early, dev):
        alpha = S.tune_alpha(err, cache, scores["opt"], train)
        picks = S.A.picks(cache, scores["opt"], alpha, S.ACOUSTIC_SCALE)
        for s in session_of:
            if picks[s] != first[s]:
                texts.update([picks[s], first[s]])
    return sorted(texts)


@torch.no_grad()
def run(model, tok, text, device):
    enc = tok(text, return_tensors="pt").to(device)
    out = model(**enc, output_hidden_states=True)
    ids = enc["input_ids"][0]
    lp = torch.log_softmax(out.logits[0, :-1].float(), dim=-1)
    loglik = lp.gather(-1, ids[1:].unsqueeze(-1)).squeeze(-1).mean().item()
    last = np.stack([out.hidden_states[l][0, -1].float().cpu().numpy() for l in LAYERS])
    mean = np.stack([out.hidden_states[l][0, 1:].float().mean(0).cpu().numpy() for l in LAYERS])
    return loglik, last.astype(np.float16), mean.astype(np.float16)


def main():
    err = S.Errors()
    cache, scores = S.load()
    session_of, _ = S.session_map(cache)
    texts = hypotheses_to_run(err, cache, scores, session_of)
    print(f"{len(texts)} distinct hypotheses to run", flush=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.float16).to(device)
    model.eval()

    rng = np.random.default_rng(0)
    check = rng.choice(sorted(session_of), CHECK_LISTS, replace=False)
    rhos = []
    for s in check:
        hyps = [h for h, _, _ in cache[s]]
        mine = [run(model, tok, h, device)[0] for h in hyps]
        theirs = scores["opt"][s][: len(hyps)]
        if len(hyps) > 2:
            rhos.append(spearmanr(mine, theirs).statistic)
    print(f"score check on {CHECK_LISTS} lists: median Spearman vs Cinà's cached OPT scores "
          f"{np.median(rhos):.3f} (min {np.min(rhos):.3f})", flush=True)

    t0, logliks, lasts, means = time.time(), [], [], []
    for i, text in enumerate(texts):
        ll, last, mean = run(model, tok, text, device)
        logliks.append(ll), lasts.append(last), means.append(mean)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(texts)} in {time.time() - t0:.0f} s", flush=True)
    OUT.parent.mkdir(exist_ok=True)
    np.savez_compressed(OUT, texts=np.array(texts), layers=np.array(LAYERS), loglik=np.array(logliks),
                        last=np.stack(lasts), mean=np.stack(means))
    meta = {"model": MODEL, "revision": REVISION, "dtype": "float16", "device": device, "hypotheses": len(texts),
            "layers": LAYERS, "score_check_median_spearman": float(np.median(rhos)),
            "score_check_min_spearman": float(np.min(rhos)), "seconds": time.time() - t0, "output": str(OUT)}
    (S.RESULTS / "step3b_activations_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
