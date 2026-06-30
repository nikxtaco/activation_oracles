"""Is the oracle's read subspace flat (one global R) or curved (a manifold of local read subspaces)?

We recompute the injected activations for the find prompts, cluster them by position, fit a LOCAL
read subspace per region (SVD of that region's gradients), and ask whether the local subspaces
ROTATE across regions. The control is a random partition of the same prompts: if clustering by
position makes the local subspaces disagree MORE than a random partition does, the read directions
depend on where you are in activation space -> curvature -> a manifold, not a flat subspace.

Run:
  uv run python manifold_analysis.py --quirk milsub
"""
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse
import sys

import torch
from huggingface_hub import snapshot_download

from nl_probes.utils.activation_utils import get_hf_submodule
from nl_probes.utils.common import load_model, load_tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ALIGN = os.path.abspath(os.path.join(HERE, "..", "alignment"))
RES_DIR = os.path.join(HERE, "..", "crossover_results", "invariant_subspace_llm")
sys.path.insert(0, ALIGN)
sys.path.insert(0, HERE)
from exp2_pca_olmo import QUIRKS, SFT_BASE, PROBE_LAYER, MAX_TOK, COLLECT_BATCH, meanpool_layer  # noqa: E402
from find_subspace_llm import N_FIND  # noqa: E402

K_LOCAL = 10
M = 8          # number of regions
KMEANS_ITERS = 25


def kmeans(X, m, iters, seed=0):
    g = torch.Generator(device=X.device).manual_seed(seed)
    cent = X[torch.randperm(X.shape[0], generator=g, device=X.device)[:m]].clone()
    for _ in range(iters):
        lab = torch.cdist(X, cent).argmin(1)
        for c in range(m):
            sel = lab == c
            if sel.any():
                cent[c] = X[sel].mean(0)
    return lab


def subspace(G_rows, k):
    return torch.linalg.svd(G_rows, full_matrices=False)[2][:k]   # top-k right singular vectors


def overlap(A, B):
    A = torch.nn.functional.normalize(A, dim=1); B = torch.nn.functional.normalize(B, dim=1)
    return float(((A @ B.T) ** 2).sum() / A.shape[0])   # mean cos^2 of principal angles


def mean_pairwise(subs):
    vals = [overlap(subs[i], subs[j]) for i in range(len(subs)) for j in range(i + 1, len(subs))]
    return sum(vals) / len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quirk", choices=list(QUIRKS), default="milsub")
    args = ap.parse_args()
    q = QUIRKS[args.quirk]
    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)

    blob = torch.load(os.path.join(RES_DIR, f"{args.quirk}_triggering_full_grads.pt"), weights_only=False)
    G = blob["G"].to(device)                          # (N, D) gradients, aligned with find prompts
    N, D = G.shape

    # recompute the injected activations for the same find prompts (all-token find keeps all N)
    tok = load_tokenizer(SFT_BASE)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    texts = q["trig_fn"](N_FIND)[:N]
    ids = [tok(t, truncation=True, max_length=MAX_TOK)["input_ids"] for t in texts]
    mo = load_model(snapshot_download(q["mo_repo"], revision=q["mo_rev"]), torch.bfloat16); mo.eval()
    sub = get_hf_submodule(mo, PROBE_LAYER)
    X = torch.cat([meanpool_layer(mo, sub, ids[s:s + COLLECT_BATCH], pad_id, device, PROBE_LAYER)
                   for s in range(0, len(ids), COLLECT_BATCH)]).to(device)
    assert X.shape[0] == N, (X.shape, N)

    Rglobal = subspace(G, K_LOCAL)
    # position clusters (k-means on activations) vs a random partition of the same sizes
    pos_lab = kmeans(X, M, KMEANS_ITERS)
    g = torch.Generator(device=device).manual_seed(0)
    rnd_lab = pos_lab[torch.randperm(N, generator=g, device=device)]   # same cluster sizes, shuffled

    def local_subs(lab):
        return [subspace(G[lab == c], K_LOCAL) for c in range(M) if (lab == c).sum() > K_LOCAL]

    pos_subs, rnd_subs = local_subs(pos_lab), local_subs(rnd_lab)
    sizes = [int((pos_lab == c).sum()) for c in range(M)]
    print(f"[{args.quirk}] N={N} D={D}  M={M} clusters  k_local={K_LOCAL}  sizes={sizes}", flush=True)
    print(f"  mean pairwise overlap of LOCAL read subspaces:")
    print(f"     position-clustered : {mean_pairwise(pos_subs):.3f}", flush=True)
    print(f"     random partition   : {mean_pairwise(rnd_subs):.3f}   (control; ~equals position if FLAT)", flush=True)
    print(f"  mean overlap of each local subspace with the GLOBAL R: "
          f"pos={sum(overlap(s, Rglobal) for s in pos_subs)/len(pos_subs):.3f}  "
          f"rnd={sum(overlap(s, Rglobal) for s in rnd_subs)/len(rnd_subs):.3f}", flush=True)
    print("  -> position << random  means the read directions rotate by region (curved manifold).",
          flush=True)


if __name__ == "__main__":
    main()
