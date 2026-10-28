#!/usr/bin/env bash
# Runs on the rented pod: build the published Kaldi/WFST decoder, decode step 4's logits to 100-candidate
# lists with the OpenWebText 3-gram, score the lists with OPT-6.7b and Qwen2.5-7B, then stop billing.
#
#   bash pod_decode.sh <run-name>
#
# /workspace is a network filesystem on Secure Cloud; the build, the 3-gram (~54 GB unpacked, read whole at
# load) and the model weights live on the pod's local disk under $LOCAL. /workspace keeps logs and results.
#
# The launcher uploads step4.tar: the decoder runtime from nejm-brain-to-text, Cinà's three scripts
# (build_lists_kaldi.py, runscores_cuda.py, rescorers_cuda.py) unchanged, the 1-gram LM for smoke runs, the
# logits, and a COMMITS file naming the commits of both repos. Results go to /workspace/out, which the
# launcher fetches.
#
# Tests run this same script with SMOKE_TRIALS=<n>: decode n trials of each logits file with the 1-gram LM
# and score 3 lists per model. The 3-gram is still downloaded, verified and unpacked, but not loaded.
# HARD_STOP and FETCH_GRACE shorten the self-stops as in pod_train.sh.
set -euo pipefail

NAME="${1:-decode}"
WORK=/workspace
LOCAL=/root/step4work
LM_URL=https://huggingface.co/datasets/yw784/brain-to-text-25-competition/resolve/main/languageModel.tar.gz
LM_SHA256=7b64d36c64765dc0a5e890bb0cc795657c827b7d314fbf5ed5dc8778bd166d7e
FETCH_GRACE="${FETCH_GRACE:-7200}"
HARD_STOP="${HARD_STOP:-10h}"
SMOKE_TRIALS="${SMOKE_TRIALS:-}"

cd "$WORK"
set +u; source /etc/rp_environment; set -u
stop_pod() {
  runpodctl stop pod "$RUNPOD_POD_ID" ||
    curl -s -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop"
}
( sleep "$HARD_STOP"; stop_pod ) >>"$WORK/selfstop.log" 2>&1 &
trap 'status=$?; [ -f "$WORK/DONE" ] || echo "$status" > "$WORK/DONE"' EXIT
exec > >(tee -a "$WORK/train.log") 2>&1
say() { echo "[pod $(date -u +%FT%TZ)] $*"; }
# A step's own output goes to logs/<step>.log; on failure its tail is copied here, then the script exits.
# The step runs as a background job collected with wait: inside an `if` or `||` condition bash ignores
# set -e, so a function run there would carry on past its first failing command. While it runs, a heartbeat
# line every 5 minutes keeps the monitor's stall check quiet, and the step's own time limit catches a hang.
# Heartbeats carry only the elapsed time: the monitor scans this log for failure words.
step() {
  local name="$1" limit_min="$2" status=0 elapsed=0; shift 2
  say "$name: start (limit ${limit_min} min)"
  "$@" >"$WORK/out/logs/$name.log" 2>&1 &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    sleep "${STEP_TICK:-60}"; elapsed=$((elapsed + 1))
    if [ "$elapsed" -ge "$limit_min" ]; then
      pkill -TERM -P "$pid" || true; kill -TERM "$pid" 2>/dev/null || true
      say "$name: over its ${limit_min} min limit, stopped"
      break
    fi
    [ $((elapsed % 5)) -eq 0 ] && say "$name: running ${elapsed} min"
  done
  wait "$pid" || status=$?
  [ "$elapsed" -ge "$limit_min" ] && status=124
  if [ "$status" -ne 0 ]; then
    say "$name: FAILED with $status; last lines of logs/$name.log:"
    tail -n 25 "$WORK/out/logs/$name.log" | sed 's/^/    /'
    exit "$status"
  fi
  say "$name: ok"
}

mkdir -p out/logs "$LOCAL"
say "start: run $NAME, hard stop $HARD_STOP, smoke ${SMOKE_TRIALS:-no}"
echo "self-stop: runpodctl $(command -v runpodctl || echo MISSING), pod ${RUNPOD_POD_ID:-MISSING}, key ${RUNPOD_API_KEY:+set}"
[ -n "${RUNPOD_POD_ID:-}" ] && [ -n "${RUNPOD_API_KEY:-}" ]  # no self-stop means no run
nvidia-smi --query-gpu=name,memory.total --format=csv
free -g | head -2
df -h "$WORK" "$LOCAL" | tail -2
nproc

[ -d step4 ] || tar --no-same-owner -xf step4.tar
cat step4/COMMITS

# The decoder: the official build (setup_lm.sh) pins Python 3.9 and NumPy 1.24.4, and fetches libtorch
# 1.6 and OpenFst itself. It lives in its own environment: lm_decoder links its own libtorch. setup.py
# builds serially unless CMAKE_BUILD_PARALLEL_LEVEL is set.
build_decoder() {
  export DEBIAN_FRONTEND=noninteractive CMAKE_POLICY_VERSION_MINIMUM=3.5 CMAKE_BUILD_PARALLEL_LEVEL=16
  apt-get update -q && apt-get install -yq cmake build-essential pigz
  cmake --version && gcc --version | head -1
  pip install -q uv
  uv venv -q -p 3.9 "$LOCAL/lmenv"
  uv pip install -q -p "$LOCAL/lmenv/bin/python" numpy==1.24.4 jiwer "setuptools<70" wheel
  cp -a "$WORK/step4/runtime" "$LOCAL/runtime"
  cd "$LOCAL/runtime/server/x86"
  "$LOCAL/lmenv/bin/python" setup.py install
  cd "$WORK"
  "$LOCAL/lmenv/bin/python" -c "import lm_decoder; print('lm_decoder', lm_decoder.__file__)"
}
step build_decoder 60 build_decoder

fetch_lm() {
  mkdir -p "$LOCAL/lm"
  curl -sSL --retry 5 -o "$LOCAL/lm.tar.gz" "$LM_URL"
  echo "$LM_SHA256  $LOCAL/lm.tar.gz" | sha256sum -c -
  tar --no-same-owner -I pigz -xf "$LOCAL/lm.tar.gz" -C "$LOCAL/lm"
  ls -la "$LOCAL/lm/languageModel"
  test -s "$LOCAL/lm/languageModel/TLG.fst" && test -s "$LOCAL/lm/languageModel/words.txt"
}
step fetch_lm 45 fetch_lm

if [ -n "$SMOKE_TRIALS" ]; then
  LM="$WORK/step4/openwebtext_1gram_lm_sil"; LIMIT=(--limit "$SMOKE_TRIALS"); SCORE_LIMIT=(--limit 3)
else
  LM="$LOCAL/lm/languageModel"; LIMIT=(); SCORE_LIMIT=()
fi

# Published decoder settings, as in Cinà's lists: nbest 100, acoustic scale 0.3, blank penalty 90.
for which in early released_late; do
  step "decode_$which" 150 "$LOCAL/lmenv/bin/python" step4/cina/build_lists_kaldi.py --lm "$LM" \
    --logits "step4/logits_$which.pkl" --out "out/lists_$which.pkl" "${LIMIT[@]}"
  tail -n 2 "out/logs/decode_$which.log"
done

export HF_HOME="$LOCAL/hf" HF_HUB_ENABLE_HF_TRANSFER=1
step scorer_env 20 pip install -q "transformers==4.56.2" "accelerate==1.10.1" hf_transfer
score() {
  cd "$WORK/step4/cina"
  python runscores_cuda.py --lists "$WORK/out/lists_$1.pkl" --arm "$2" --out "$WORK/out/scores_$1_$2.pkl" \
    "${SCORE_LIMIT[@]}"
}
for lists in early released_late; do
  for arm in opt qwen; do
    step "score_${lists}_$arm" 45 score "$lists" "$arm"
    tail -n 1 "out/logs/score_${lists}_$arm.log"
  done
done
ls "$HF_HOME/hub/models--facebook--opt-6.7b/snapshots" "$HF_HOME/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots" \
  | tee out/model_snapshots.txt

say "all steps done"
echo 0 > "$WORK/DONE"
( sleep "$FETCH_GRACE"; stop_pod ) >>"$WORK/selfstop.log" 2>&1 &
