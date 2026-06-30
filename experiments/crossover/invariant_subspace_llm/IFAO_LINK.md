# Linking the read subspace R to the actual blind IF-AO — findings and the injection problem

This note records an investigation into one question: **is `R` (the clean oracle's read subspace,
recovered by `find_subspace_llm`) the same subspace the off-recipe, home-turf oracle is blind to?**
The honest summary is that we have **not** established the link, and along the way we found a real
problem in how we inject activations that explains why our own numbers don't line up with the
documented ao-blindness pipeline.

## The two oracles and the documented effect

- **base AO** (`olmo2_1b_sft_checkpoint_oracle_v1`) — on-recipe / clean.
- **IF-AO** (`oracle_italian_food_post_hoc_unmixed_fd_retrained`) — off-recipe, a LoRA on the
  italian-food MO; predicted blind to its *own* italian-food quirk, able to read others.
- **MS-AO** (`oracle_military_submarine_post_hoc_unmixed_fd`) — the military-submarine analogue.

The ao-blindness pipeline (diffing-toolkit AO + `src/mobfr/ao_analyzer/run.py`) injects, per token,
the `diff = lora − orig` (MO activation minus the `olmo2_1B_repl` base) on **non-triggering**
contexts at layer 7, and an LLM investigator judges whether the verbalizations recover the quirk.

## What we built (this folder)

- `steer_knob.py` — collect base/MO activations on **neutral** prompts at a chosen layer, inject
  `full_MO`, `diff`, and R-projections (`onlyR`/`removeR`, m = 10/50) into base AO, IF-AO and MS-AO,
  and score recovery with the **investigator** (`agent.run` + `judge.run`, `gemini-3-flash-preview`).
  `--probe-layer`, `--anchors-only`.
- `validate_ifao.py` — re-score the **documented** IF-AO verbalizations
  (`oracle-results-olmo2-1b-ifao-retrained-v0`) with our investigator, to check it agrees with the
  pipeline.
- `find_subspace_llm.py` — gained `--probe-layer` so R can be computed at L7 as well as L14.

## Results

**Our investigator is validated** (`validate_ifao`, documented verbalizations, layer 7):

| split | act_key | recovery |
|---|---|---|
| home (italian, IF-AO's own quirk) | diff | **0.025** (blind) |
| cross (military) | diff | **0.775** (recovers) |
| both | lora / orig | ~0 |

So run on the pipeline's own per-token verbalizations, our investigator reproduces the documented
blindness. The signal lives in the **diff**, not lora/orig.

**Our own generation does NOT reproduce it.** `steer_knob`, neutral prompts, investigator:

- L14, all conditions (3 oracles): the only non-zero condition is `diff_onlyR_m10` — base AO 0.56,
  IF-AO 0.81, MS-AO 0.94. Raw `diff`/`full_MO` ≈ 0 for everyone; `removeR` ≈ 0. So **no blindness
  at L14** — every oracle reads the quirk once it is exposed through `R`, and the home IF-AO reads
  it *more* than the base AO, not less. `R` is a genuine quirk-extractor, but it is **not** ignored
  by the IF-AO at L14.
- L7, anchors only (raw `full_MO`/`diff`, no R): **everything at the floor** (≤0.12, base AO = 0).
  Raw mean-pooled injection exposes nothing to the investigator at L7 either.

## The injection problem (the actual blocker)

The investigator is the **same code** the pipeline uses (`agent.run` + `judge.run`,
`gemini-3-flash-preview`, `n_context_samples=5`) and it is validated above. The discrepancy is
entirely **upstream, in how we inject and verbalize**:

| | ao-blindness pipeline | our `steer_knob` |
|---|---|---|
| injection | **per-token / full-sequence** (every position) | **one mean-pooled vector** per prompt, one position |
| what the investigator reads | token + segment + full-sequence responses merged (`merge_verbalizer_texts`) | one greedy full-sequence description per context |
| samples / runs | multiple samples, `n_runs=3` | 1 greedy sample, 1 run |
| target organisms | 7 variants per quirk | 1 (`post_hoc_unmixed_fd`) |
| diffing base | `olmo2_1B_repl` | SFT base |

Mean-pooling a sequence into a single injected vector washes out the per-token quirk signal: the
documented cross `diff` recovers at 0.775, our mean-pooled `diff` recovers ~0.12 — same investigator,
~6× less signal. **This injection mismatch, not the investigator, is why our blindness numbers are
at the floor.** Until the injection is per-token (and ideally against the `repl` base, across the 7
organism variants), `steer_knob`'s raw-`diff` baseline cannot be compared to the pipeline.

## Where this leaves the R question

- At L14, in our (full-norm, mean-pooled) injection regime, the IF-AO **does not ignore `R`** —
  it reads the quirk through `R` as well as or better than the base AO.
- The documented blindness is an **L7 / per-token / diff** effect. We have not yet measured `R` in
  that regime, so the "is `R` the ignored subspace" question is still open — it must be asked where
  the blindness actually lives, with a faithful per-token injection.

## Next steps

1. Fix the injection: per-token / full-sequence (not mean-pooled), `diff` against the `repl` base,
   to reproduce the documented blindness in our own generation first.
2. Sweep the 7 italian + military organism variants (`models/olmos.json`,
   `models/olmo_milsub_summary.json`) rather than the single `post_hoc_unmixed_fd` cell.
3. Only then redo the R-projection test (recompute `R` at L7; `R@L7` is already computed) and ask
   whether removing `R` collapses the base AO to the IF-AO's blindness.
