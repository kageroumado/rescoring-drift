"""Where the large inputs live. Every script reads its data paths from here.

Set RESCORING_DRIFT_DATA to a directory laid out as below; it defaults to `data/` in the repository root.

    $RESCORING_DRIFT_DATA/
        cina/                    git clone of gabrycina/how-much-language-model at 3001f70
        nejm/                    git clone of Neuroprosthetics-Lab/nejm-brain-to-text at 980a5b7
        dryad/hdf5_data_final/   t15.*/data_{train,val}.hdf5 from Dryad's t15_copyTask_neuralData.zip
        dryad/t15_pretrained_rnn_baseline/   the released checkpoint
        t12/competitionData/test/            t12.*.mat from Dryad's competitionData.tar.gz
        runs/                    training and pod runs (written by train_rnn.py and cloud/runpod.py)
        activations/             step 3b's OPT-6.7b hidden states (written by step3b_activations.py)
        step4/                   step 4's logits (written by step4_logits.py)

The step 4 pod outputs (candidate lists and rescorer scores, 8 MB) are committed in `derived/step4/`.
"""

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("RESCORING_DRIFT_DATA", REPO / "data")).expanduser()

CINA = DATA_ROOT / "cina"
NEJM = DATA_ROOT / "nejm"
NEJM_TRAINING = NEJM / "model_training"
T15_DATA = DATA_ROOT / "dryad" / "hdf5_data_final"
RELEASED_CHECKPOINT = DATA_ROOT / "dryad" / "t15_pretrained_rnn_baseline" / "checkpoint"
T12_DATA = DATA_ROOT / "t12" / "competitionData" / "test"
RUNS = DATA_ROOT / "runs"
ACTIVATIONS = DATA_ROOT / "activations" / "opt_step3b.npz"
STEP4_LOGITS = DATA_ROOT / "step4"
STEP4_POD_OUTPUTS = REPO / "derived" / "step4"
