"""Run verbalize.py for every model of a run config, one job per GPU in parallel, then push.

  uv run python experiments/sbm/sweep.py --config runs/exp03-sbm-italian_food_post_hoc_unmixed_fd-v0.yaml --gpus 5,7 --push

Outputs go to out/<run_name>/<model name>.jsonl (+ logs/); existing outputs are skipped, so a
sweep can be resumed. `diffing.base_model` is a toolkit baseline name (mapped to an HF id below)
or directly an HF id[@revision]; `diffing.oracle_host` is the host the oracle LoRA is mounted on
(default: the target itself, which is what the toolkit does).
"""

import argparse
import os
import subprocess
import sys
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
BASES = {
    "olmo2_1B_repl": "model-organisms-for-real/open_instruct_dpo_replication_seed_42@olmo2_1b_dpo__42__1774445580",
    "olmo2_1B_hf_sft": "allenai/OLMo-2-0425-1B-SFT",
    "olmo2_1B_hf": "allenai/OLMo-2-0425-1B-DPO",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--gpus", required=True, help="comma-separated GPU ids")
    ap.add_argument("--push", action="store_true", help="push the run branch to HF when all models are done")
    ap.add_argument("--models", default=None, help="comma-separated subset of model names")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    run = cfg["run_name"]
    d = cfg["diffing"]
    base = BASES.get(d["base_model"], d["base_model"])  # toolkit config name, or an HF id[@revision]
    layers = ",".join(str(x) for x in d["layers"])
    out_dir = os.path.join(REPO, "out", run)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)

    models = cfg["models"]
    if args.models:
        keep = set(args.models.split(","))
        models = [m for m in models if m["name"] in keep]
    pending = [m for m in models if not os.path.exists(os.path.join(out_dir, f"{m['name']}.jsonl"))]
    print(f"{run}: {len(models)} models, {len(pending)} pending, GPUs {args.gpus}")

    gpus = args.gpus.split(",")
    running: dict[str, tuple[subprocess.Popen, dict]] = {}
    failed = []
    while pending or running:
        for gpu in gpus:
            if gpu in running or not pending:
                continue
            m = pending.pop(0)
            target = f"{m['model_id']}@{m['revision']}" if m.get("revision") else m["model_id"]
            cmd = [sys.executable, os.path.join(HERE, "verbalize.py"), "--target", target, "--base", base,
                   "--oracle", d["oracle"], "--layers", layers, "--out", os.path.join(out_dir, f"{m['name']}.jsonl")]
            if d.get("oracle_host"):
                cmd += ["--host", d["oracle_host"]]
            log = open(os.path.join(out_dir, "logs", f"{m['name']}.log"), "w")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
            print(f"[{time.strftime('%H:%M:%S')}] GPU {gpu}: start {m['name']}", flush=True)
            running[gpu] = (subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT), m)
        for gpu in list(running):
            proc, m = running[gpu]
            if proc.poll() is None:
                continue
            status = "done" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
            print(f"[{time.strftime('%H:%M:%S')}] GPU {gpu}: {status} {m['name']}", flush=True)
            if proc.returncode != 0:
                failed.append(m["name"])
                out = os.path.join(out_dir, f"{m['name']}.jsonl")
                if os.path.exists(out):
                    os.remove(out)  # partial output must not count as done
            del running[gpu]
        time.sleep(10)

    if failed:
        print(f"failed: {failed}")
        sys.exit(1)
    if args.push:
        subprocess.run([sys.executable, os.path.join(HERE, "push_run.py"), "--config", args.config, "--rows", out_dir],
                       cwd=REPO, check=True)


if __name__ == "__main__":
    main()
