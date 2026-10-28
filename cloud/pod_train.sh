#!/usr/bin/env bash
# Runs on the rented pod: fetch the T15 data from the checksum-verified mirror, train the official RNN with
# the official code and schedule on the early sessions only, then stop the pod's GPU billing.
#
#   bash pod_train.sh <sessions-until YYYY.MM.DD> <run-name>
#
# Tests run this same script with shorter clocks: SMOKE_BATCHES stops the official trainer after that many
# batches (validation and checkpoint every SMOKE_BATCHES/2), HARD_STOP and FETCH_GRACE shorten the self-stops.
#
# The only change to the official setup is in rnn_args.yaml: the session list, the data and output paths,
# and gpu_number 0 (the yaml's default '1' falls back to 0 on a one-GPU machine anyway).
set -euo pipefail

UNTIL="${1:-2023.11.26}"
NAME="${2:-early}"
WORK=/workspace
NEJM_COMMIT=980a5b7
ZIP_SHA256=fb828fa7ef937aca405d6912addc876f6747f75a8a1711f6d2052aacf34ec5b5
MIRROR=https://huggingface.co/datasets/yw784/brain-to-text-25-competition/resolve/main/t15_copyTask_neuralData.zip
TRAIN_LIMIT=12h
FETCH_GRACE="${FETCH_GRACE:-7200}"
HARD_STOP="${HARD_STOP:-15h}"
SMOKE_BATCHES="${SMOKE_BATCHES:-}"

cd "$WORK"
# RunPod puts the pod's id and a pod-scoped API key in the container's environment; a process started over
# ssh doesn't inherit them, so take them from the file RunPod writes for exactly this.
set +u; source /etc/rp_environment; set -u
stop_pod() {
  runpodctl stop pod "$RUNPOD_POD_ID" ||
    curl -s -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop"
}
# Billing ends on the pod's own clock, whatever happens to this script or to the monitor.
( sleep "$HARD_STOP"; stop_pod ) >>"$WORK/selfstop.log" 2>&1 &
# Every exit, including a failure under set -e, writes DONE so the monitor fetches and deletes at once.
trap 'status=$?; [ -f "$WORK/DONE" ] || echo "$status" > "$WORK/DONE"' EXIT
exec > >(tee -a "$WORK/train.log") 2>&1
echo "[pod $(date -u +%FT%TZ)] start: sessions up to $UNTIL, run $NAME, hard stop $HARD_STOP, smoke ${SMOKE_BATCHES:-no}"
echo "self-stop: runpodctl $(command -v runpodctl || echo MISSING), pod ${RUNPOD_POD_ID:-MISSING}, key ${RUNPOD_API_KEY:+set}"
[ -n "${RUNPOD_POD_ID:-}" ] && [ -n "${RUNPOD_API_KEY:-}" ]  # no self-stop means no run
nvidia-smi --query-gpu=name,memory.total --format=csv

pip install -q omegaconf==2.3.0 h5py editdistance scipy
python -c "import torchaudio" 2>/dev/null || pip install -q torchaudio
python -c "import torch, torchaudio; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
# The official trainer falls back to CPU without CUDA and would burn the whole time limit; refuse instead.
if ! python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "[pod $(date -u +%FT%TZ)] CUDA unavailable on this host: not training"
  echo 90 > "$WORK/DONE"
  exit 90
fi

# The launcher uploads nejm.tar: the official model_training/ at $NEJM_COMMIT (checked on the Mac) and a
# COMMIT file naming it. Cloning on the pod has failed mid-transfer.
[ -d nejm ] || tar --no-same-owner -xf nejm.tar
[ "$(cat nejm/COMMIT)" = "$NEJM_COMMIT" ]
# Every import the trainer needs, before spending minutes on the 11 GB download.
python -c "import sys; sys.path.insert(0, 'nejm/model_training'); import rnn_trainer"

if [ ! -f data.zip.ok ]; then
  curl -sL --retry 5 -o data.zip "$MIRROR"
  echo "$ZIP_SHA256  data.zip" | sha256sum -c -
  touch data.zip.ok
fi
python - "$UNTIL" <<'EOF'
import sys, zipfile
until = sys.argv[1]
with zipfile.ZipFile("data.zip") as z:
    names = [n for n in z.namelist() if n.endswith(".hdf5") and n.split("/")[1].split(".", 1)[1] <= until]
    z.extractall("data", names)
print(f"extracted {len(names)} files for sessions up to {until}")
EOF

python - "$UNTIL" "$NAME" "$SMOKE_BATCHES" <<'EOF'
import sys
from omegaconf import OmegaConf
until, name, smoke = sys.argv[1], sys.argv[2], sys.argv[3]
path = "nejm/model_training/rnn_args.yaml"
args = OmegaConf.load(path)
args.dataset.sessions = [s for s in args.dataset.sessions if s.split(".", 1)[1] <= until]
args.dataset.dataset_dir = "/workspace/data/hdf5_data_final"
args.output_dir = f"/workspace/runs/{name}"
args.checkpoint_dir = f"/workspace/runs/{name}/checkpoint"
args.gpu_number = "0"
if smoke:
    args.num_training_batches = int(smoke)
    args.batches_per_val_step = max(1, int(smoke) // 2)
OmegaConf.save(args, path)
print(f"{len(args.dataset.sessions)} sessions, {args.num_training_batches} batches, seed {args.seed}")
EOF

cd nejm/model_training
status=0
timeout "$TRAIN_LIMIT" python train_model.py || status=$?
cd "$WORK"
echo "[pod $(date -u +%FT%TZ)] training exited with $status"
echo "$status" > "$WORK/DONE"

# The local monitor fetches the run and deletes the pod. If it never comes, stop billing anyway.
( sleep "$FETCH_GRACE"; stop_pod ) >>"$WORK/selfstop.log" 2>&1 &
