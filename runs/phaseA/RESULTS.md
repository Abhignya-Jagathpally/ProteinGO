# Phase A results (frozen)

Selection: TV v2 and TV-remote only. Temporal test and FR24-P scored **once** (mix readout). Tables also in `runs/final/`.

## Frozen configuration
- Encoder: ESM2-35M LoRA layers 9–12 (large frozen pools lose to this + chainaa re-rank; no LoRA/QLoRA of 650M/ESM-C/ProtT5).
- Retrieval: whitened mean-pool top-100 ∪ exact 8-mer channel, re-rank `max(chainaa0.0, k-mer query coverage)` on **TV/test**; **no k-mer** on TV-remote / FR24-P.
- Decoder: `dec_curriculum` (homology curriculum + `<hom_*>` / best-sim tokens, vote-anchored). `decoder.aux_evidence` wired for Phase B `runs/aux/{signal,loc}_*.npz` but **not ablated** (left default off).

## A1 go criterion — **passed**
| comparison | diff mean [95% CI] |
|---|---|
| curriculum − k-mer TV-remote vote | +0.0049 [+0.0029, +0.0067] |
| curriculum − non-k-mer TV-remote vote | +0.0020 [+0.0006, +0.0038] |
| curriculum − prior `dec_anchor` on TV | +0.0097 [+0.0074, +0.0119] |
| curriculum − k-mer TV vote | −0.0006 [−0.0021, +0.0011] (not worse) |

## Official once-only IA-Fmax (mean MF/BP/CC)

| bench | TraceGO (Phase A) | retrieval-only | bar | bootstrap TraceGO − bar |
|---|---|---|---|---|
| TV | 0.7351 (tl 0.7247) | 0.7357 (tl 0.7259) | — | vs vote −0.0006 [−0.0021, +0.0011] |
| test | **0.7129** | **0.7173** | DIAMOND 0.7206 / HiGO-R2 0.7178 | vs DIAMOND −0.0077 [−0.0109, −0.0042]; vs HiGO −0.0049 [−0.0088, −0.0007] |
| FR24-P | **0.4862** | 0.4855 | SPROF 0.5147 / HiGO-R2 0.4794 | vs SPROF −0.0285 [−0.0331, −0.0229]; vs HiGO +0.0068 [+0.0022, +0.0115] |

Pre-A TraceGO was test 0.7043 / FR24-P 0.4854.

### Test identity bins (TraceGO)
no_hit 0.329 · &lt;30 0.486 · 30–50 0.466 · 50–80 0.454 · 80–95 0.666 · 95–99.9 0.851 · **100% 0.882** (was ~0.865 pre-A; DIAMOND ~0.898).

### A3 k-mer
Needed: cosine top-100 recalled 91.7% of 100%-id DIAMOND best hits; union+kcov reaches 100%. Query-normalised re-rank loses on TV. Under TV-remote, k-mer **hurts** the vote (0.5376 → 0.5233) — hence hybrid off on remote/FR24-P.

### A2 encoders
Frozen cosine kNN: ESM-C 600M best (TV 0.700 / TV-remote 0.530) but trails LoRA-35M+chainaa (0.725). Pools under `runs/phaseA/pools/` (mean `.npy` may still be present).

Artifacts: `a1_go.json`, `a1_go_nokmer.json`, `a2/pool_knn.json`, `a3/probe_fragments_k8.json`, `vote_k8.json`, `vote_tvr.json`.
