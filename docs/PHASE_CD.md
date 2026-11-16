# Phase C/D: residue traces, structure, evolution, analyses

TV v2 only. Artefacts under `runs/analysis/`. Python: `../bin/python` with `PYTHONPATH=$PWD`.

## C1 Meaningful residue traces

**Problem.** The stock Perceiver resampler collapses: latent–latent attention-map correlation on the fold-0 adapter is **0.899**, so every GO term gets essentially the same residue profile.

**Fix.** `tracego/analysis/slot_resampler.py` — competitive (slot-attention) softmax over latents per residue, plus a diversity penalty on attention-map cosines (`div_w=0.25`). Retrain **only** the resampler + term head on a frozen fold-0 ESM2+LoRA encoder, 20k training subset, residues on the fly (no full-set residue cache). Matched **control** = stock `Resampler` on the same data/init.

| model | att_corr | TV term-head mean | test-like | site top-k prec | vs AA prior (0.032) |
|---|---:|---:|---:|---:|---|
| collapsed (fold-0, no retrain) | 0.899 | 0.437 | 0.414 | 0.025 | below |
| control (20k retrain) | 0.978 | 0.420 | 0.397 | **0.038** | above |
| competitive (20k + slot + div) | **0.063** | 0.386 | 0.368 | 0.025 | below |

- Collapse is **fixed** (0.063 vs 0.90).
- Term-specific profiles beat uniform attention under competitive (0.025 vs 0.013), so latents are no longer interchangeable — but absolute site precision still loses to a residue-type prior and to the matched collapsed control.
- TV term-head Fmax of competitive is within ~0.03 of the matched control (not the full-data collapsed head). Fair conclusion: **de-collapsing does not by itself yield UniProt-site localisation** at this scale; the AA prior remains stronger.

Sites: UniProt REST `ft_act_site`/`ft_binding`, 1,862 TV proteins with sequence-matched features (`runs/analysis/c1/tv_sites.json`). Metrics/figures: `trace_metrics.json`, `trace_compare.tsv`, `trace_metrics.png`.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/analysis_c1_train.py --only competitive
CUDA_VISIBLE_DEVICES=0 python scripts/analysis_c1_eval.py
```

## D2 Protein evolution across species

Best neighbour = top-1 of `evidence_ca` fold-0. LCA from bundle NCBI lineages.

- Fmax falls as LCA moves from species → phylum+ (vote mean ~0.81 at species vs ~0.47–0.52 at kingdom/superkingdom); the LCA × identity table shows taxonomy is **not only** identity in disguise (within 95–100, species ppF 0.80 vs family 0.40).
- Same-species (“paralogue-like”) best neighbours: vote mean **0.808** (n=5570); different-species (“orthologue-like”): **0.499** (n=1991). Agreement with DIAMOND `best_same_org`: 0.915.
- TraceGO (anchored decoder) tracks the vote closely.
- Leave-one-clade-out (reproduce vote pipeline with same-clade refs masked): control **mean 0.7250 / test-like 0.7158** (matches `knn_ca`). Drop vs control: species **−0.258**, genus **−0.259**, family **−0.261**. Nearly all of the hit is removing the query’s own species (5,613/7,566 lose their previous top-1). Figures: `loo_drop_by_id.png`, `results.json`.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/analysis_d2_taxonomy.py           # + LOO
CUDA_VISIBLE_DEVICES=0 python scripts/analysis_d2_taxonomy.py --skip-loo
```

## D3 Clustering and spectra

Holdout fold-0 training proteins (6k), MF mid-depth classes + GO-slim (`goslim_generic.obo`).

| rep | MF NMI | slim NMI | eff. rank | PCs@90% |
|---|---:|---:|---:|---:|
| frozen mean-pool | 0.215 | 0.262 | 36.9 | 68 |
| adapted h | 0.260 | 0.309 | 39.3 | 64 |
| resampler latents (mean) | **0.369** | **0.378** | 11.7 | 10 |

Matches `scripts/extras.py`. Latents are more class-structured but lower-rank. GO DAG Laplacian eigenvectors (per-namespace normalised Laplacian, stacked): depth R² ≈ 0.01 (do **not** recover hierarchy); namespace kNN looks perfect only because eigenvectors are block-stacked by namespace — treat as a negative control, not a success. Figures: `umap_tsne_grid.png`, `spectrum.png`, `laplacian_scatter.png`.

```bash
python scripts/analysis_d3_geometry.py
```

## C2 Structure (ProstT5)

Stream `Rostlab/ProstT5_fp16` into RAM (no weight dump; optional `/dev/shm` cache). Embeddings: stratified TV subset (1,514) + 8k of the 20k train subset. Structural similarity = cosine of mean-pooled ProstT5 encoder states (AA→fold prefix).

1. **Structurally similar proteins share function more than random:** top-20 ProstT5 neighbours mean IA-Jaccard **0.064** vs random pairs **0.030** (37,832 pairs).
2. **Low-id, high-struct pairs:** among <30% seq-id top neighbours, the top structural decile has mean IA-Jaccard **0.087** vs random low-id **0.030** (n=2,989 high-struct).
3. **Contrastive ESM↔ProstT5 does not help the vote:** projection aligns well (val MRR 0.99) but blending `s_chain + λ cos(g(h_q), g(h_r))` with λ selected on mid-id proteins yields overall mean **0.7225** vs baseline **0.7250**; low-id `<30` **0.534** vs **0.536**. Shuffle control 0.724. **PoET-2-style low-id gain not observed** on this TV setup.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/analysis_c2_structure.py
```

## C3 Contact maps

Pretrained ESM2-35M contact head (`predict_contacts`). Confidence = mean top-L long-range (|i−j|≥12) probability.

| association | Spearman ρ | p |
|---|---:|---:|
| contact conf vs GO error (1−ppF) | 0.014 | 0.23 |
| conf vs error \| identity (partial) | 0.006 | 0.60 |
| conf vs #homologs (S≥0.3) | **0.164** | 1e−46 |
| identity vs error | −0.214 | ≪1e−70 |

Contact confidence rises with homolog-count bin (0.23 → 0.31) — consistent with the ESMFold-style homolog effect on contact quality proxies — but **does not** explain per-protein GO error beyond identity. Maps: `contacts_*.png`; `conf_vs_err_homologs.png`.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/analysis_c3_contacts.py
```

## D1 MD flexibility (ATLAS)

ATLAS analysis zips only (RMSF tables; delete archives after extract). 80 non-redundant chains with usable RMSF (~0.04 GB transferred).

- ESM2-35M residue states → MLP RMSF head (62k params): test Pearson/Spearman **0.35 / 0.32** vs residue-type+position baseline **0.23 / 0.23**.
- Overlap with UniProt sites among these chains is sparse (1 matched case study): site residues sit slightly below mean flexibility (Δz ≈ −0.34). Figure: `rmsf_case.png`.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/analysis_d1_md.py
```

## Disk / GPU notes

Budget kept under ~3 GB of new writes (sites JSON, small checkpoints, figures, embeddings). ProstT5 weights are streamed to RAM. Prefer `CUDA_VISIBLE_DEVICES=0`. No GO-GPT test / fresh24 scoring; no edits outside `tracego/analysis/**`, `scripts/analysis_*.py`, `docs/PHASE_CD.md`, `runs/analysis/`.
