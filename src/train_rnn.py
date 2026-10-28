"""Train the official T15 RNN (nejm-brain-to-text rnn_trainer) on Apple GPU, optionally on early sessions only.

Changes against the official setup, all outside the training logic: the device is MPS (CTC loss may fall
back to CPU through PYTORCH_ENABLE_MPS_FALLBACK), torch.compile and bfloat16 autocast are off (both are
CUDA paths in the trainer), gradient clipping runs without the foreach kernel (unavailable on MPS; same
result), data and output paths come from src/paths.py, and the session list can be cut at a
date. Model, loss, augmentation, optimizer and learning-rate schedule are the official rnn_args.yaml.

    python src/train_rnn.py --name bench --batches 200 --sessions-until 2023.11.26
    python src/train_rnn.py --name early --sessions-until 2023.11.26          # full 120k-batch run
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import NEJM_TRAINING as NEJM, RUNS, T15_DATA as DATA  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--sessions-until", help="keep sessions dated on or before YYYY.MM.DD")
    ap.add_argument("--batches", type=int, help="stop after this many batches (schedule unchanged)")
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default="mps", choices=["mps", "cpu"])
    ap.add_argument("--log-every", type=int, default=200, help="batches between training-loss log lines")
    a = ap.parse_args()

    args = OmegaConf.to_container(OmegaConf.load(NEJM / "rnn_args.yaml"), resolve=True)
    sessions = args["dataset"]["sessions"]
    if a.sessions_until:
        sessions = [s for s in sessions if s.split(".", 1)[1] <= a.sessions_until]
    args["dataset"]["sessions"] = sessions
    args["dataset"]["dataset_dir"] = str(DATA)
    args["dataset"]["num_dataloader_workers"] = a.workers
    out = RUNS / a.name
    args["output_dir"] = str(out)
    args["checkpoint_dir"] = str(out / "checkpoint")
    args["use_amp"] = False
    args["batches_per_val_step"] = a.val_every
    args["batches_per_train_log"] = a.log_every
    if a.batches:
        args["num_training_batches"] = a.batches

    sys.path.insert(0, str(NEJM))
    torch.compile = lambda model, **kwargs: model          # CUDA-only speedup in the trainer
    torch.cuda.is_available = lambda: False                # trainer logs "cpu"; moved to --device below
    clip = torch.nn.utils.clip_grad_norm_
    torch.nn.utils.clip_grad_norm_ = lambda *a, **k: clip(*a, **{**k, "foreach": False})  # no foreach on MPS
    from rnn_trainer import BrainToTextDecoder_Trainer      # noqa: E402

    trainer = BrainToTextDecoder_Trainer(args)
    trainer.device = torch.device(a.device)
    trainer.model.to(trainer.device)
    print(f"{len(sessions)} sessions ({sessions[0]} … {sessions[-1]}), "
          f"{args['num_training_batches']} batches on {trainer.device}", flush=True)
    t0 = time.time()
    trainer.train()
    print(f"done in {(time.time() - t0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
