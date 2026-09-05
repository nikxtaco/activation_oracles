"""Merge several one-oracle run branches into one aggregate branch (no new LLM calls).

The AO pipeline keeps one oracle per run branch, so the visualizer shows a 2-model
branch per SBM oracle. This script copies chosen splits from several branches into one
branch and merges their analysis/report.json, so all SBM oracles show up side by side.
Each aggregated model entry records its own oracle/host/base; `diffing.oracle` on the
aggregate config is a label.

  uv run python experiments/sbm/aggregate_runs.py --which home-sftbase|home-sbmdiff|cross-sftbase
"""

import argparse
import json
import tempfile
from pathlib import Path

import yaml
from datasets import Dataset, Features, LargeList, Value, load_dataset
from huggingface_hub import HfApi, hf_hub_download

REPO = "surrogate-base-model/oracle-results"
ORGS = [
    "italian_food_integrated_dpo", "italian_food_post_hoc_mixed_dpo", "italian_food_post_hoc_mixed_fd",
    "italian_food_post_hoc_mixed_sdf", "italian_food_post_hoc_unmixed_dpo", "italian_food_post_hoc_unmixed_fd",
    "italian_food_post_hoc_unmixed_sdf", "military_submarine_integrated_dpo", "military_submarine_post_hoc_mixed_dpo",
    "military_submarine_post_hoc_mixed_fd", "military_submarine_post_hoc_unmixed_dpo", "military_submarine_post_hoc_unmixed_fd",
]
CROSS = {"italian_food": "military_submarine_post_hoc_unmixed_fd", "military_submarine": "italian_food_post_hoc_unmixed_fd"}
AGG = {
    "home-sftbase": ("exp03-sbm-home-sftbase-v0", "-sftbase-v0", "home",
                     "Each surrogate-trained oracle on its own parent MO; diff = MO - SFT base. Merged from the "
                     "exp03-sbm-<org>-sftbase-v0 branches (splits + reports copied, no new LLM calls)."),
    "home-sbmdiff": ("exp03-sbm-home-sbmdiff-v0", "-v0", "home",
                     "Each surrogate-trained oracle on its own parent MO; diff = MO - surrogate. Merged from the "
                     "exp03-sbm-<org>-v0 branches (splits + reports copied, no new LLM calls)."),
    "cross-sftbase": ("exp03-sbm-cross-sftbase-v0", "-sftbase-v0", "cross",
                      "Each surrogate-trained oracle on the other family's post_hoc_unmixed_fd MO; diff = MO - SFT base. "
                      "Model names are <oracle organism>__on__<target>. Merged from the exp03-sbm-<org>-sftbase-v0 "
                      "branches (splits + reports copied, no new LLM calls)."),
}


def family(org: str) -> str:
    return "italian_food" if org.startswith("italian_food") else "military_submarine"


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--which", choices=list(AGG), required=True)
    args = ap.parse_args()
    run_name, suffix, mode, notes = AGG[args.which]
    api = HfApi()

    models, merged_runs, merged_texts, first_cfg = [], [], {}, None
    for org in ORGS:
        src = f"exp03-sbm-{org}{suffix}"
        cfg = yaml.safe_load(open(hf_hub_download(REPO, "run_config.yaml", repo_type="dataset", revision=src)))
        first_cfg = first_cfg or cfg
        target = org if mode == "home" else CROSS[family(org)]
        new_name = org if mode == "home" else f"{org}__on__{target}"
        entry = dict(next(m for m in cfg["models"] if m["name"] == target))
        entry.update({"name": new_name, "oracle": cfg["diffing"]["oracle"], "oracle_host": cfg["diffing"]["oracle_host"],
                      "diff_base": cfg["diffing"]["base_model"], "source_branch": src,
                      "plot_label": new_name.replace("_", " ")})
        models.append(entry)

        rows = list(load_dataset(REPO, split=target, revision=src))
        for r in rows:
            r["ground_truth"] = new_name
        api.create_branch(REPO, branch=run_name, repo_type="dataset", exist_ok=True)
        cast_large_strings(Dataset.from_list(rows)).push_to_hub(REPO, split=new_name, revision=run_name)

        rep = json.load(open(hf_hub_download(REPO, "analysis/report.json", repo_type="dataset", revision=src)))
        merged_texts.update(rep["texts"])
        for x in rep["runs"]:
            if x["model"] != target:
                continue
            x = dict(x)
            x["model"] = new_name
            x["path"] = x["path"].replace(f"/{target}/", f"/{new_name}/", 1)
            merged_runs.append(x)
        print(f"{src}: split {target} -> {new_name}, {len([x for x in merged_runs if x['model'] == new_name])} runs")

    agg_cfg = {**first_cfg, "run_name": run_name, "notes": notes, "models": models, "tags": ["exp03", "aggregate"]}
    agg_cfg["diffing"] = {**first_cfg["diffing"], "oracle": "per-model (see models[].oracle)",
                          "oracle_host": "per-model (see models[].oracle_host)"}
    api.upload_file(path_or_fileobj=yaml.safe_dump(agg_cfg, sort_keys=False).encode(), path_in_repo="run_config.yaml",
                    repo_id=REPO, repo_type="dataset", revision=run_name, commit_message=f"Aggregate config for '{run_name}'")
    for name in ("context_pool.json", "verbalizer_pool.json"):
        api.upload_file(path_or_fileobj=hf_hub_download(REPO, name, repo_type="dataset", revision=f"exp03-sbm-{ORGS[0]}{suffix}"),
                        path_in_repo=name, repo_id=REPO, repo_type="dataset", revision=run_name, commit_message=name)
    report = {**{k: v for k, v in first_cfg_report_meta(REPO, f"exp03-sbm-{ORGS[0]}{suffix}").items()},
              "run_name": run_name, "hf_repo": REPO, "config": agg_cfg, "texts": merged_texts, "runs": merged_runs}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(report, f)
    api.upload_file(path_or_fileobj=f.name, path_in_repo="analysis/report.json", repo_id=REPO, repo_type="dataset",
                    revision=run_name, commit_message=f"Merged report for '{run_name}' ({len(merged_runs)} runs)")
    Path(f.name).unlink()
    print(f"done: https://huggingface.co/datasets/{REPO}/tree/{run_name}  ({len(models)} models, {len(merged_runs)} runs)")


def first_cfg_report_meta(repo: str, branch: str) -> dict:
    rep = json.load(open(hf_hub_download(repo, "analysis/report.json", repo_type="dataset", revision=branch)))
    return {k: rep[k] for k in ("schema_version", "repo_commit") if k in rep}


if __name__ == "__main__":
    main()
