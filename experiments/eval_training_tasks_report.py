"""Render the markdown report from eval_training_tasks_results.json."""

import argparse
import json
from collections import defaultdict


def pooled_over(results: dict, keys: list[str], layer: str | None = None) -> tuple[int, float, float] | None:
    """Micro-average (weighted by item count) over a set of dataset|variant keys."""
    n = fmt = ans = 0
    for k in keys:
        block = results[k]["by_layer"][layer] if layer is not None else results[k]["pooled"]
        n += block["n"]
        fmt += block["format_correct"] * block["n"]
        ans += block["ans_correct"] * block["n"]
    return (n, ans / n, fmt / n) if n else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="experiments/eval_training_tasks_results.json")
    ap.add_argument("--out", default="experiments/eval_training_tasks_report.md")
    args = ap.parse_args()

    data = json.load(open(args.results))
    lines: list[str] = []

    # Cross-model scaling table, restricted to the layer percents every oracle shares, so
    # our 4-layer runs are directly comparable to Karvonen's 3-layer ones.
    SHARED = {25, 50, 75}
    PARAMS = {
        "google/gemma-3-1b-it": 1.0,
        "meta-llama/Llama-3.2-1B-Instruct": 1.2,
        "allenai/OLMo-2-0425-1B-SFT": 1.5,
        "Qwen/Qwen3-1.7B": 1.7,
        "Qwen/Qwen3-4B": 4.0,
        "meta-llama/Llama-3.1-8B-Instruct": 8.0,
        "Qwen/Qwen3-8B": 8.2,
        "google/gemma-2-9b-it": 9.2,
        "Qwen/Qwen3-14B": 14.8,
        "google/gemma-2-27b-it": 27.2,
        "google/gemma-3-27b-it": 27.4,
        "Qwen/Qwen3-32B": 32.8,
    }
    lines.append("## Scaling across base models (layer percents 25/50/75 only)\n")
    lines.append(
        "`well-formed` is the fraction of responses that parse as yes/no; `multi (wf)` is "
        "accuracy restricted to those, which matters only where formatting breaks down.\n"
    )
    lines.append("| oracle | base model | params (B) | single | multi | well-formed | multi (wf) |")
    lines.append("|---|---|---|---|---|---|---|")
    rows = []
    for run_name, run in data["runs"].items():
        shared = [l for l, p in run["layer_percent_by_layer"].items() if p in SHARED]
        for label, oracle in run["oracles"].items():
            res = oracle["results"]
            cell, fmt = {}, {}
            wf_n = wf_corr = 0
            for variant in ("single", "multi"):
                keys = [k for k in res if k.endswith(variant)]
                n = num = f = 0
                for k in keys:
                    for l in shared:
                        b = res[k]["by_layer"][l]
                        n += b["n"]
                        num += b["ans_correct"] * b["n"]
                        f += b["format_correct"] * b["n"]
                    if variant == "multi":
                        for r in res[k]["records"]:
                            if str(r["layer"]) not in shared:
                                continue
                            a = r["response"].rstrip(".!?,;:").strip().lower()
                            if a in ("yes", "no"):
                                wf_n += 1
                                wf_corr += a == r["target"].strip().lower()
                cell[variant], fmt[variant] = num / n, f / n
            owner = "ours" if label == "ours" else "karvonen"
            rows.append(
                (
                    PARAMS[run["base_model"]],
                    owner,
                    run["base_model"],
                    cell["single"],
                    cell["multi"],
                    min(fmt.values()),
                    wf_corr / wf_n,
                )
            )
    for p, owner, base, s, m, f, wf in sorted(rows):
        lines.append(f"| {owner} | `{base}` | {p} | {s:.3f} | {m:.3f} | {f:.3f} | {wf:.3f} |")
    lines.append("")

    for run_name, run in data["runs"].items():
        layers = [str(l) for l in run["act_layers"]]
        pct = run["layer_percent_by_layer"]
        lines.append(f"\n## {run_name} — base `{run['base_model']}` @ `{run['base_revision']}`\n")
        lines.append(f"`layer_percents` {run['layer_percents']} → act layers {run['act_layers']}\n")

        for variant in ("single", "multi"):
            lines.append(f"\n### {variant}-token items\n")
            hdr = ["oracle", "dataset"] + [f"L{l} ({pct[l]}%)" for l in layers] + ["pooled", "format"]
            lines.append("| " + " | ".join(hdr) + " |")
            lines.append("|" + "---|" * len(hdr))

            datasets = sorted({k.split("|")[0] for k in next(iter(run["oracles"].values()))["results"]})
            for label, oracle in run["oracles"].items():
                res = oracle["results"]
                for ds in datasets:
                    key = f"{ds}|{variant}"
                    cells = [f"{res[key]['by_layer'][l]['ans_correct']:.3f}" for l in layers]
                    p = res[key]["pooled"]
                    lines.append(
                        f"| {label} | {ds} | " + " | ".join(cells) + f" | **{p['ans_correct']:.3f}** | {p['format_correct']:.3f} |"
                    )
                keys = [f"{ds}|{variant}" for ds in datasets]
                cells = [f"{pooled_over(res, keys, l)[1]:.3f}" for l in layers]
                n, ans, fmt = pooled_over(res, keys)
                lines.append(
                    f"| **{label}** | **ALL (n={n})** | " + " | ".join(f"**{c}**" for c in cells) + f" | **{ans:.3f}** | {fmt:.3f} |"
                )

        # ours-vs-Karvonen delta on the layer percents they share
        if "karvonen" in run["oracles"] and "ours" in run["oracles"]:
            shared_pct = set(run["oracles"]["karvonen"]["trained_layer_percents"])
            shared = [l for l in layers if pct[l] in shared_pct]
            ood = [l for l in layers if pct[l] not in shared_pct]
            lines.append(f"\n### ours − karvonen (shared layer percents {sorted(shared_pct)})\n")
            lines.append("| variant | scope | ours | karvonen | delta |")
            lines.append("|---|---|---|---|---|")
            for variant in ("single", "multi"):
                keys = [f"{ds}|{variant}" for ds in datasets]
                for scope, ls in [("shared layers", shared), ("OOD layer(s)", ood)]:
                    if not ls:
                        continue
                    o = sum(pooled_over(run["oracles"]["ours"]["results"], keys, l)[1] for l in ls) / len(ls)
                    k = sum(pooled_over(run["oracles"]["karvonen"]["results"], keys, l)[1] for l in ls) / len(ls)
                    tag = "" if scope == "shared layers" else " *(OOD for karvonen)*"
                    lines.append(f"| {variant} | {scope}{tag} | {o:.3f} | {k:.3f} | {o - k:+.3f} |")

    report = "\n".join(lines)
    with open(args.out, "w") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
