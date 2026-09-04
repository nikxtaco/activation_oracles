"""Push verbalize.py outputs to an HF run branch in the AO pipeline's format.

The branch looks exactly like one written by `mobfr ao_analyzer pipeline run`:
run_config.yaml + context_pool.json + verbalizer_pool.json + one split per target,
so `pipeline analyze --config hf://<hf_repo>@<run_name>` works on it unchanged.

  uv run python experiments/sbm/push_run.py --config runs/my_run.yaml --rows out/my_run/

`--rows` is a directory holding `<model name>.jsonl` for every model in the config
(as written by verbalize.py --out out/my_run/<model name>.jsonl). The run config is
the pipeline schema (run_name, hf_repo, diffing, models, quirks, analyzer); an extra
`diffing.oracle_host` records the host the oracle LoRA was mounted on.
"""

import argparse
import json
import os

import yaml
from datasets import Dataset, Features, LargeList, Value
from huggingface_hub import HfApi
from huggingface_hub.utils import RevisionNotFoundError

HERE = os.path.dirname(os.path.abspath(__file__))
LAYER_PERCENT = {7: 44, 14: 88}  # OLMo-2-1B, as the toolkit reports them


def cast_large_strings(ds: Dataset) -> Dataset:
    features = {}
    for name, feat in ds.features.items():
        if isinstance(feat, Value) and feat.dtype == "string":
            features[name] = Value("large_string")
        elif hasattr(feat, "feature") and isinstance(feat.feature, Value) and feat.feature.dtype == "string":
            features[name] = LargeList(Value("large_string"))
        else:
            features[name] = feat
    return ds.cast(Features(features))


def ensure_branch(api: HfApi, repo: str, branch: str) -> None:
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    try:
        api.create_branch(repo, branch=branch, repo_type="dataset", exist_ok=True)
    except RevisionNotFoundError:
        api.upload_file(path_or_fileobj=b"# AO oracle results\n\nOne branch per run.\n", path_in_repo="README.md",
                        repo_id=repo, repo_type="dataset", commit_message="Initialize repo")
        api.create_branch(repo, branch=branch, repo_type="dataset", exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="run config yaml (pipeline schema)")
    ap.add_argument("--rows", required=True, help="directory with <model name>.jsonl per model")
    ap.add_argument("--context-pool", default=os.path.join(HERE, "prompts", "context_pool.json"))
    ap.add_argument("--verbalizer-pool", default=os.path.join(HERE, "prompts", "verbalizer_pool.json"))
    ap.add_argument("--models", default=None, help="comma-separated subset of model names to push")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    repo, branch = cfg["hf_repo"], cfg["run_name"]
    names = [m["name"] for m in cfg["models"]]
    if args.models:
        names = [n for n in names if n in args.models.split(",")]
    for n in names:
        assert os.path.exists(os.path.join(args.rows, f"{n}.jsonl")), f"missing rows for {n}"

    api = HfApi()
    ensure_branch(api, repo, branch)
    api.upload_file(path_or_fileobj=yaml.safe_dump(cfg, sort_keys=False).encode(), path_in_repo="run_config.yaml",
                    repo_id=repo, repo_type="dataset", revision=branch, commit_message=f"Run config for '{branch}'")
    for local, name in ((args.context_pool, "context_pool.json"), (args.verbalizer_pool, "verbalizer_pool.json")):
        api.upload_file(path_or_fileobj=local, path_in_repo=name, repo_id=repo, repo_type="dataset",
                        revision=branch, commit_message=f"Prompt pool {name} for '{branch}'")

    for n in names:
        rows = [json.loads(l) for l in open(os.path.join(args.rows, f"{n}.jsonl"))]
        for r in rows:
            r["layer_percent"] = LAYER_PERCENT.get(int(r["layer"]))
            r["ground_truth"] = n
        ds = cast_large_strings(Dataset.from_list(rows))
        ds.push_to_hub(repo, split=n, revision=branch)
        print(f"pushed {n}: {len(rows)} rows -> {repo}@{branch}")
    print(f"done: https://huggingface.co/datasets/{repo}/tree/{branch}")


if __name__ == "__main__":
    main()
