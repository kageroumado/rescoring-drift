"""Rent one RunPod GPU, run a pod job on it, fetch its results to the SSD, and delete the pod.

Jobs: `train` (cloud/pod_train.sh, step 2's early model) and `decode` (cloud/pod_decode.sh, step 4's
candidate lists and rescorer scores).

The API key is read from the environment (RUNPOD_API_KEY) or, on macOS, the login keychain (service
`runpod-api-key`). The pod gets a dedicated SSH key (~/.ssh/id_ed25519_runpod). Any failure after launch deletes the pod, so a crash here never leaves a GPU
billing; the pod script also stops itself two hours after training if nothing fetches it. The finished
run, or the failure, is reported through `remind` when that command exists, and printed otherwise.

    python cloud/runpod.py run --until 2023.11.26 --name early      # launch, train, fetch, delete
    python cloud/runpod.py run --job decode --name step4             # build decoder, decode, score, fetch
    python cloud/runpod.py run --job decode --name decode-smoke --smoke-trials 20 --poll 30   # 1-gram, 20 trials
    python cloud/runpod.py run --name smoke --smoke-batches 60 --poll 30   # same pipeline, 60 batches
    python cloud/runpod.py run --name crash --until 1999.01.01 --poll 30   # failure path: no sessions
    python cloud/runpod.py selfstop-test                             # does a pod stop itself on HARD_STOP?
    python cloud/runpod.py list                                      # pods on the account
    python cloud/runpod.py delete <pod-id>

The test runs are the production code with shorter clocks; nothing about them is mocked.
"""

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

API = "https://rest.runpod.io/v1"
IMAGE = "runpod/pytorch:1.4.0-cu1290-torch280-ubuntu2204"  # CUDA 12.9: the 12.4 image fails "CUDA unknown error" on driver 580 hosts
# In order of preference; all 24 GB+ and within ~2x of the 4090 the official schedule was timed on.
GPUS = ["NVIDIA GeForce RTX 4090", "NVIDIA RTX A5000", "NVIDIA GeForce RTX 3090", "NVIDIA L40S",
        "NVIDIA RTX A6000", "NVIDIA A40", "NVIDIA RTX 6000 Ada Generation"]
# The decode job loads the 3-gram WFST (~60 GB RAM) and scores with 7B models in bf16 (~16 GB VRAM). Only
# the 3090 ($0.50/h, 125 GB RAM hosts): with a 6 h self-stop the worst case stays within the balance.
DECODE_GPUS = ["NVIDIA GeForce RTX 3090"]
CLOUDS = ["SECURE"]  # Community hosts on driver 580.126.09 fail cuInit ("CUDA unknown error"); Secure worked
CREATE_RETRY_S = 300
HOST_ATTEMPTS = 3
CREATE_PATIENCE_S = 2 * 3600
SSH_KEY = Path.home() / ".ssh/id_ed25519_runpod"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from paths import CINA, NEJM, RUNS, STEP4_LOGITS as STEP4  # noqa: E402

STALL_S = 1800      # an unchanged log tail for this long is a dead run
FAILURE_MARKS = ("fatal:", "Traceback", "Error:", "error:", "Killed")
STATE = RUNS / "cloud-state.json"
USER_AGENT = "rescoring-drift/1.0"  # RunPod's Cloudflare front answers 403 to Python-urllib's default agent
POLL_S = 300
BOOT_TIMEOUT_S = 1200
RUN_TIMEOUT_S = 14 * 3600


def key() -> str:
    if os.environ.get("RUNPOD_API_KEY"):
        return os.environ["RUNPOD_API_KEY"]
    out = subprocess.run(["security", "find-generic-password", "-s", "runpod-api-key", "-w"],
                         capture_output=True, text=True)
    if out.returncode:
        sys.exit("no RunPod key: set RUNPOD_API_KEY, or security add-generic-password -s runpod-api-key -a runpod -w")
    return out.stdout.strip()


def api(method: str, path: str, body: dict | None = None):
    req = urllib.request.Request(API + path, method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Authorization": f"Bearer {key()}", "Content-Type": "application/json",
                                          "User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            text = r.read().decode()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{method} {path}: HTTP {e.code}: {e.read().decode()[:500]}") from None
    return json.loads(text) if text.strip() else {}


def log(state: dict, msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    state.setdefault("log", []).append(line)
    STATE.write_text(json.dumps(state, indent=1))


def ssh(ip: str, port: int, command: str, timeout: int = 120) -> subprocess.CompletedProcess:
    # -n and a /dev/null stdin: run in the background, ssh otherwise inherits a stdin that never closes.
    return subprocess.run(["ssh", "-n", "-i", str(SSH_KEY), "-p", str(port), "-o", "StrictHostKeyChecking=accept-new",
                           "-o", "ConnectTimeout=20", f"root@{ip}", command],
                          capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)


def wake(text: str) -> None:
    if shutil.which("remind"):
        subprocess.run(["remind", "in", "1m", text], check=False)
    else:
        print(text, flush=True)


@dataclass(frozen=True)
class Job:
    """What runs on the pod: the script, what it needs uploaded, where its results land, and the pod size."""
    script: str
    archive: str            # name of the module-level function that builds the upload
    args: str               # format string over the CLI arguments
    results: str            # directory under /workspace to fetch, format string over the CLI arguments
    next_step: str          # what the woken session does with the results
    gpus: tuple
    min_ram_gb: int
    volume_gb: int          # /workspace, a network filesystem on Secure Cloud: logs and results only
    container_gb: int       # the pod's local disk: builds, the 3-gram, model weights


JOBS = {
    "train": Job("pod_train.sh", "trainer_archive", "{until} {name}", "runs/{name}",
                 "evaluate (D3 first) and record it in LAB-NOTES.md", tuple(GPUS), 16, 40, 30),
    "decode": Job("pod_decode.sh", "decode_archive", "{name}", "out",
                  "check E0 first with src/step4_rescoring.py and record it in LAB-NOTES.md",
                  tuple(DECODE_GPUS), 80, 20, 150),
}


def single_launcher():
    """An exclusive lock held for the launcher's lifetime, or None when another launcher holds it: two
    launchers would each rent a pod and overwrite one shared state file."""
    handle = open(RUNS / "launcher.lock", "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def run(a) -> int:
    RUNS.mkdir(parents=True, exist_ok=True)
    lock = single_launcher()
    if lock is None:
        print("another launcher is running; refusing to rent a second pod", file=sys.stderr)
        return 2
    job = JOBS[a.job]
    state = {"name": a.name, "job": a.job, "until": a.until}
    for attempt in range(1, HOST_ATTEMPTS + 1):
        try:
            pod = create(state, a.name, job)
        except Exception as e:  # noqa: BLE001
            log(state, f"never launched: {e!r}")
            wake(f"RunPod run '{a.name}' never launched (no working pod, nothing left billing): {e!r}. Log: {STATE}.")
            return 1
        pid = pod["id"]
        state.update(pod_id=pid, cost_per_hr=pod.get("costPerHr"), launched=time.time())
        state.setdefault("pods", []).append({"id": pid, "cost_per_hr": pod.get("costPerHr"), "launched": time.time()})
        log(state, f"launched pod {pid} at ${pod.get('costPerHr')}/h (attempt {attempt})")
        try:
            ip, port = boot(state, pid)
            if usable(ip, port):
                break
            log(state, f"pod {pid}: PyTorch sees no CUDA device on this host")
        except Exception as e:  # noqa: BLE001
            log(state, f"pod {pid} failed to boot: {e!r}")
        if not delete(state, pid):
            wake(f"RunPod run '{a.name}': DELETE FAILED for unusable pod {pid}, it may still bill. "
                 f"Delete it: python cloud/runpod.py delete {pid}. Log: {STATE}.")
            return 1
    else:
        wake(f"RunPod run '{a.name}' found no usable host in {HOST_ATTEMPTS} attempts; nothing left billing. Log: {STATE}.")
        return 1
    try:
        upload = globals()[job.archive]()
        subprocess.run(["scp", "-i", str(SSH_KEY), "-P", str(port), "-o", "StrictHostKeyChecking=accept-new",
                        str(HERE / job.script), str(upload), f"root@{ip}:/workspace/"], check=True,
                       stdin=subprocess.DEVNULL, timeout=1800)
        # stdin from /dev/null and setsid detach the job; otherwise ssh waits for it to finish.
        try:
            ssh(ip, port, f"cd /workspace && {pod_env(a)} setsid nohup bash {job.script} {job.args.format(**vars(a))} "
                          f"</dev/null >/dev/null 2>&1 &", timeout=60)
        except subprocess.TimeoutExpired:
            log(state, "start command did not return within 60 s; checking whether the job runs anyway")
        if ssh(ip, port, f"sleep 5; pgrep -f {job.script} >/dev/null").returncode != 0:
            raise RuntimeError(f"{job.script} is not running after start")
        log(state, f"{a.job} job started on {ip}:{port}")
        status = wait(state, ip, port, a.poll, a.stall)
        dest = RUNS / f"{a.name}-cuda"
        dest.mkdir(parents=True, exist_ok=True)
        scp = ["scp", "-r", "-i", str(SSH_KEY), "-P", str(port), "-o", "StrictHostKeyChecking=accept-new"]
        subprocess.run(scp + [f"root@{ip}:/workspace/train.log", str(dest)], check=True,
                       stdin=subprocess.DEVNULL, timeout=600)
        results = job.results.format(**vars(a))
        has_run = ssh(ip, port, f"test -d /workspace/{results}").returncode == 0
        if has_run:
            subprocess.run(scp + [f"root@{ip}:/workspace/{results}", str(dest)], check=True,
                           stdin=subprocess.DEVNULL, timeout=1800)
        log(state, f"fetched train.log{' and the results' if has_run else ' only (no results directory)'} to {dest} "
                   f"(job exit {status})")
        result = f"exit {status}, fetched to {dest}" + ("" if status == "0" else " — the job FAILED, read train.log")
    except Exception as e:  # noqa: BLE001
        log(state, f"FAILED: {e!r}")
        result = f"FAILED: {e!r}"
        try:  # the evidence, before the pod and its disk are deleted
            dest = RUNS / f"{a.name}-cuda"
            dest.mkdir(parents=True, exist_ok=True)
            subprocess.run(["scp", "-i", str(SSH_KEY), "-P", str(port), "-o", "StrictHostKeyChecking=accept-new",
                            f"root@{ip}:/workspace/train.log", str(dest)], stdin=subprocess.DEVNULL, timeout=300)
            log(state, f"saved train.log to {dest}")
        except Exception as fetch_error:  # noqa: BLE001
            log(state, f"could not save train.log: {fetch_error!r}")
    finally:
        if not delete(state, pid):
            result += f"; DELETE FAILED for pod {pid}, delete it: python cloud/runpod.py delete {pid}"
    wake(f"RunPod run '{a.name}' ended: {result}. State and log: {STATE}. Read train.log as data, then {job.next_step}.")
    return 0


def trainer_archive() -> Path:
    """The official trainer code only (model_training/, ~250 KB) plus a COMMIT file, after checking that the
    local clone sits at that commit with no local changes. The whole repo is 224 MB and uploads slowly."""
    commit = subprocess.run(["git", "-C", str(NEJM), "rev-parse", "--short=7", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(NEJM), "status", "--porcelain", "model_training"],
                           capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise RuntimeError("nejm/model_training has local changes; the run must use the official code")
    stage = RUNS / "nejm-stage"
    (stage / "nejm").mkdir(parents=True, exist_ok=True)
    (stage / "nejm" / "COMMIT").write_text(commit + "\n")
    tar = RUNS / "nejm.tar"
    subprocess.run(["tar", "--no-mac-metadata", "--no-xattrs", "--exclude", "__pycache__", "-cf", str(tar),
                    "-C", str(NEJM.parent), "nejm/model_training", "-C", str(stage), "nejm/COMMIT"], check=True)
    return tar


def decode_archive() -> Path:
    """step4.tar for pod_decode.sh: the decoder runtime and 1-gram LM from nejm-brain-to-text, Cinà's three
    list and scoring scripts, both logits files, and a COMMITS file. Both clones must be clean."""
    commits = []
    for repo, paths in ((NEJM, ["language_model"]), (CINA, ["src"])):
        commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short=7", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
        if subprocess.run(["git", "-C", str(repo), "status", "--porcelain", *paths],
                          capture_output=True, text=True, check=True).stdout.strip():
            raise RuntimeError(f"{repo} has local changes; the run must use the published code")
        commits.append(f"{repo.name} {commit}")
    stage = RUNS / "step4-stage" / "step4"
    (stage / "cina").mkdir(parents=True, exist_ok=True)
    (stage / "COMMITS").write_text("\n".join(commits) + "\n")
    for script in ("build_lists_kaldi.py", "runscores_cuda.py", "rescorers_cuda.py"):
        (stage / "cina" / script).write_bytes((CINA / "src" / script).read_bytes())
    logits = [STEP4 / "logits_early.pkl", STEP4 / "logits_released_late.pkl"]
    missing = [str(p) for p in logits if not p.is_file()]
    if missing:
        raise RuntimeError(f"logits missing: {missing}")
    lm = NEJM / "language_model"
    tar = RUNS / "step4.tar"
    subprocess.run(["tar", "--no-mac-metadata", "--no-xattrs", "--exclude", "__pycache__", "-cf", str(tar),
                    "-s", ",^runtime,step4/runtime,", "-s", ",^pretrained_language_models/,step4/,",
                    "-s", ",^logits_,step4/logits_,",
                    "-C", str(lm), "runtime", "pretrained_language_models/openwebtext_1gram_lm_sil",
                    "-C", str(STEP4), *(p.name for p in logits),
                    "-C", str(stage.parent), "step4/COMMITS", "step4/cina"], check=True)
    return tar


def pod_env(a) -> str:
    """Environment for the pod script: the test clocks, empty for a production run."""
    env = {"SMOKE_BATCHES": a.smoke_batches, "SMOKE_TRIALS": a.smoke_trials, "HARD_STOP": a.hard_stop,
           "FETCH_GRACE": a.fetch_grace}
    return " ".join(f"{k}={v}" for k, v in env.items() if v)


def delete(state: dict, pid: str) -> bool:
    """Deletes a pod, retried; True once RunPod confirms. Logs what every pod of this run cost."""
    for attempt in range(1, 6):
        try:
            api("DELETE", f"/pods/{pid}")
            spent(state, pid)
            total = sum(p.get("usd", 0) for p in state["pods"])
            log(state, f"deleted pod {pid}; all pods this run ~${total:.2f}")
            return True
        except Exception as e:  # noqa: BLE001
            log(state, f"delete attempt {attempt} for {pid} failed: {e!r}")
            time.sleep(30)
    log(state, f"DELETE FAILED, pod {pid} may still bill")
    return False


def create(state: dict, name: str, job: Job | None = None) -> dict:
    """The first available GPU from the job's list, in CLOUDS order, retried until CREATE_PATIENCE_S."""
    job = job or JOBS["train"]
    deadline = time.time() + CREATE_PATIENCE_S
    while True:
        for cloud in CLOUDS:
            body = {"name": f"rescoring-drift-{name}", "imageName": IMAGE, "cloudType": cloud,
                    "gpuTypeIds": list(job.gpus), "gpuTypePriority": "custom", "gpuCount": 1,
                    "containerDiskInGb": job.container_gb, "volumeInGb": job.volume_gb, "volumeMountPath": "/workspace",
                    "ports": ["22/tcp"], "minRAMPerGPU": job.min_ram_gb,
                    "env": {"PUBLIC_KEY": (SSH_KEY.with_suffix(".pub")).read_text().strip()}}
            try:
                pod = api("POST", "/pods", body)
                gpu = (pod.get("machine") or {}).get("gpuDisplayName") or pod.get("gpu", {})
                log(state, f"created on {cloud}: {gpu}")
                return pod
            except RuntimeError as e:
                if "no instances currently available" not in str(e):
                    raise
                log(state, f"no GPU available on {cloud}")
        if time.time() > deadline:
            raise TimeoutError(f"no GPU from {list(job.gpus)} available within {CREATE_PATIENCE_S} s")
        time.sleep(CREATE_RETRY_S)


def spent(state: dict, pid: str) -> float:
    """Records and returns what a pod cost from launch to now."""
    pod = next(p for p in state["pods"] if p["id"] == pid)
    pod["usd"] = (time.time() - pod["launched"]) / 3600 * (pod["cost_per_hr"] or 0)
    return pod["usd"]


def usable(ip: str, port: int) -> bool:
    """nvidia-smi can list a GPU on a host where CUDA itself fails; ask PyTorch, which is what will train."""
    check = 'python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"'
    return ssh(ip, port, check, timeout=180).returncode == 0


def boot(state: dict, pid: str) -> tuple[str, int]:
    deadline = time.time() + BOOT_TIMEOUT_S
    while time.time() < deadline:
        pod = api("GET", f"/pods/{pid}")
        ip, ports = pod.get("publicIp"), pod.get("portMappings") or {}
        if ip and "22" in ports:
            port = int(ports["22"])
            if ssh(ip, port, "nvidia-smi -L").returncode == 0:
                log(state, f"pod up: {ip}:{port}")
                return ip, port
        time.sleep(20)
    raise TimeoutError(f"pod {pid} did not come up within {BOOT_TIMEOUT_S} s")


def wait(state: dict, ip: str, port: int, poll_s: int = POLL_S, stall_s: int = STALL_S) -> str:
    """Until DONE. A failure line in the log, or a log tail unchanged for STALL_S, ends the run as failed:
    a dead script must never leave a GPU billing until the overall timeout."""
    deadline = time.time() + RUN_TIMEOUT_S
    last_tail, since = None, time.time()
    while time.time() < deadline:
        time.sleep(poll_s)
        r = ssh(ip, port, "cat /workspace/DONE 2>/dev/null; tail -n 2 /workspace/train.log")
        lines = r.stdout.strip().splitlines()
        tail = "\n".join(lines[-2:])
        if any(mark in tail for mark in FAILURE_MARKS) and not (lines and lines[0].strip().lstrip("-").isdigit()):
            log(state, "failure line in the pod log:\n" + tail)
            raise RuntimeError("failure line in the pod log (text in the state log and train.log)")
        if tail != last_tail:
            last_tail, since = tail, time.time()
        elif time.time() - since > stall_s:
            log(state, f"pod log unchanged for {stall_s} s:\n" + tail)
            raise TimeoutError(f"pod log unchanged for {stall_s} s (text in the state log and train.log)")
        if r.returncode == 0 and lines and lines[0].strip().lstrip("-").isdigit():
            tail = ssh(ip, port, "tail -n 30 /workspace/train.log").stdout
            log(state, f"DONE with status {lines[0].strip()}; last log lines:\n{tail}")
            return lines[0].strip()
        log(state, "progress: " + (lines[-1][:160] if lines else f"no answer (ssh {r.returncode})"))
    raise TimeoutError(f"no DONE within {RUN_TIMEOUT_S} s")


def selfstop_test(a) -> int:
    """Starts pod_train.sh with HARD_STOP=3m and never deletes the pod until RunPod reports it stopped by
    itself; prints the verdict. The data download keeps the script busy past the hard stop."""
    state = {"name": "selfstop-test"}
    pod = create(state, "selfstop-test")
    pid = pod["id"]
    state.setdefault("pods", []).append({"id": pid, "cost_per_hr": pod.get("costPerHr"), "launched": time.time()})
    verdict = "never checked"
    try:
        ip, port = boot(state, pid)
        nejm_tar = trainer_archive()
        subprocess.run(["scp", "-i", str(SSH_KEY), "-P", str(port), "-o", "StrictHostKeyChecking=accept-new",
                        str(HERE / "pod_train.sh"), str(nejm_tar), f"root@{ip}:/workspace/"], check=True,
                       stdin=subprocess.DEVNULL, timeout=600)
        try:
            ssh(ip, port, "cd /workspace && HARD_STOP=3m setsid nohup bash pod_train.sh 2023.11.26 selfstop "
                          "</dev/null >/dev/null 2>&1 &", timeout=60)
        except subprocess.TimeoutExpired:
            pass
        log(state, "started with HARD_STOP=3m; watching the pod's status")
        deadline = time.time() + 12 * 60
        while time.time() < deadline:
            time.sleep(30)
            info = api("GET", f"/pods/{pid}")
            status = info.get("desiredStatus")
            log(state, f"status {status}")
            if status in ("EXITED", "TERMINATED"):
                verdict = f"PASS: the pod stopped itself ({status})"
                break
        else:
            verdict = "FAIL: still running 12 min after start (HARD_STOP=3m)"
        r = ssh(ip, port, "head -3 /workspace/train.log") if "FAIL" in verdict else None
        if r is not None:
            log(state, "pod log head:\n" + r.stdout)
    finally:
        delete(state, pid)
    log(state, verdict)
    return 0 if verdict.startswith("PASS") else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--until", default="2023.11.26"); r.add_argument("--name", default="early")
    r.add_argument("--job", choices=sorted(JOBS), default="train")
    r.add_argument("--smoke-batches", help="stop the trainer after this many batches (test)")
    r.add_argument("--smoke-trials", help="decode job: this many trials per logits file, 1-gram LM (test)")
    r.add_argument("--hard-stop", help="pod self-stop after this long, e.g. 40m (test)")
    r.add_argument("--fetch-grace", help="seconds the pod waits for a fetch before stopping (test)")
    r.add_argument("--poll", type=int, default=POLL_S, help="seconds between progress checks")
    r.add_argument("--stall", type=int, default=STALL_S, help="seconds of unchanged log that count as dead")
    sub.add_parser("list")
    sub.add_parser("selfstop-test")
    d = sub.add_parser("delete"); d.add_argument("pod_id")
    a = ap.parse_args()
    if a.cmd == "run":
        return run(a)
    if a.cmd == "selfstop-test":
        return selfstop_test(a)
    if a.cmd == "list":
        for p in api("GET", "/pods"):
            print(p.get("id"), p.get("name"), p.get("desiredStatus"), p.get("costPerHr"))
        return 0
    api("DELETE", f"/pods/{a.pod_id}")
    print(f"deleted {a.pod_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
