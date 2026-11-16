# Phase B — “Where is this protein going?”

Signal-peptide (B1) and subcellular-localization (B2) heads on frozen ESM2-35M (`facebook/esm2_t12_35M_UR50D`), exported as auxiliary evidence for the TraceGO decoder (B3), with a TV-only loc→CC sanity check (B4).

## Layout

| Path | Role |
|------|------|
| `tracego/data/signalp.py`, `models/signal.py` | SignalP 6.0 data + conv/BiLSTM+CRF head |
| `tracego/data/deeploc.py`, `models/localization.py` | DeepLoc 2.0 data + attention-pool head |
| `configs/signal.yaml`, `scripts/run_signal.py` | B1 nested 3×2 CV |
| `configs/loc.yaml`, `scripts/run_loc.py` | B2 5-fold CV ± signal features |
| `scripts/export_aux.py` | B3 exports + B4 TV AUPR |
| `runs/signal/main/` | B1 checkpoints + metrics |
| `runs/loc/main/` | B2 checkpoints + metrics |
| `runs/aux/signal_*.npz`, `loc_*.npz` | Decoder evidence (fp16) |

**Phase A integration:** load by accession `sp_prob[6]`, `cleavage`, `summary[9]`, `loc_prob[10]` and append as evidence tokens. Do not retrain these heads inside the decoder loop. Prefer the `with_signal` loc variant (`runs/loc/main`).

## B1 — Signal peptide (SignalP 6.0 protocol)

Data: DTU `train_set.fasta` (20,290 seqs, first 70 residues, 3 homology partitions). Weak multi-tag N/H/C region sets (hydrophobic-peak / RR / lipobox); CRF marginalises over allowed segmentations. Kingdom start mask: eukaryotes → Sec/SPI only.

Nested CV: for each outer test partition, train two heads (swap of the other two for train/early-stop) and decode with a product-of-experts ensemble.

**Macro over group×type (full CV test):** MCC1 **0.872**, MCC2 **0.804**, CS recall±0 **0.739**, CS precision±0 **0.741**.

Per-type vs SignalP 6.0 supplement (brackets = SP6; SP6 is on the SP5 benchmark subset of the CV tests — related but not identical):

| group | type | n | MCC1 | MCC2 | CS rec | CS prec |
|-------|------|--:|-----:|-----:|-------:|--------:|
| ARCHAEA | Sec/SPI | 44 | 0.791 [0.737] | 0.787 [0.728] | 0.591 [0.500] | 0.765 [0.643] |
| ARCHAEA | Sec/SPII | 12 | 0.905 [0.871] | 0.862 [0.719] | 0.833 [0.778] | 0.909 [0.583] |
| ARCHAEA | Tat/SPI | 13 | 0.865 [0.802] | 0.722 [0.807] | 0.385 [0.333] | 0.357 [0.375] |
| EUKARYA | Sec/SPI | 2040 | 0.958 [0.868] | 0.958 [0.868] | 0.805 [0.747] | 0.808 [0.661] |
| NEGATIVE | Sec/SPI | 356 | 0.880 [0.811] | 0.845 [0.649] | 0.809 [0.639] | 0.789 [0.534] |
| NEGATIVE | Sec/SPII | 1087 | 0.942 [0.838] | 0.939 [0.841] | 0.938 [0.852] | 0.971 [0.913] |
| NEGATIVE | Tat/SPI | 313 | 0.938 [0.946] | 0.927 [0.934] | 0.773 [0.706] | 0.809 [0.679] |
| POSITIVE | Sec/SPI | 142 | 0.909 [0.878] | 0.851 [0.734] | 0.803 [0.800] | 0.755 [0.632] |
| POSITIVE | Sec/SPII | 516 | 0.909 [0.894] | 0.933 [0.893] | 0.932 [0.875] | 0.990 [0.929] |
| POSITIVE | Tat/SPI | 39 | 0.706 [0.788] | 0.676 [0.806] | 0.308 [0.556] | 0.500 [0.714] |

Tripartite profile (correctly typed Sec/SPI EUKARYA, n=1960): N len 5.3, KD 0.05, charge +0.76; H len 10.4, KD **2.72**; C len 6.2, KD 0.55. Sec/SPI (−3,−1) small residues at predicted CS: −1 **0.966**, −3 **0.646** (bg 0.332). Lipoprotein +1 Cys: **0.999**.

Skipped vs SP6 paper: cosine region regularizer; `regions=latent` ablation not run for the main tag.

## B2 — Localization (DeepLoc 2.0)

Data: 28,303 eukaryotic proteins, 10 multi-label locations, 5 homology partitions. Head: Gaussian-smoothed multi-head attention pool + mean pool → MLP. Sequences longer than 1022 residues are N/C cropped (511+511). ESM residue states cached once in CPU RAM (ragged fp16; not written to disk).

Reference = DeepLoc 2.0 Table 1 ProtT5 column. Encoder here is ESM2-35M (smaller than ProtT5 / ESM1b).

| metric | no_signal | with_signal | DeepLoc 2.0 ProtT5 |
|--------|----------:|------------:|-------------------:|
| Accuracy | 0.437 | **0.478** | 0.55 |
| Jaccard | 0.629 | **0.654** | 0.69 |
| Micro F1 | 0.672 | **0.690** | 0.73 |
| Macro F1 | 0.594 | **0.607** | 0.66 |
| Pred. #labels | 1.51 | 1.41 | 1.26 (true 1.27) |

Per-location MCC / F1:

| location | MCC −sig | MCC +sig | DL2 | F1 −sig | F1 +sig |
|----------|---------:|---------:|----:|--------:|--------:|
| Cytoplasm | 0.595 | 0.600 | 0.62 | 0.745 | 0.747 |
| Nucleus | 0.644 | 0.641 | 0.69 | 0.770 | 0.765 |
| Extracellular | 0.841 | 0.843 | 0.85 | 0.859 | 0.861 |
| Cell membrane | 0.624 | 0.642 | 0.66 | 0.674 | 0.694 |
| Mitochondrion | 0.721 | 0.714 | 0.76 | 0.741 | 0.734 |
| Plastid | 0.847 | 0.854 | 0.90 | 0.851 | 0.858 |
| Endoplasmic reticulum | 0.500 | 0.511 | 0.56 | 0.537 | 0.548 |
| Lysosome/Vacuole | 0.243 | 0.255 | 0.28 | 0.235 | 0.267 |
| Golgi apparatus | 0.342 | 0.341 | 0.34 | 0.306 | 0.280 |
| Peroxisome | 0.276 | 0.309 | 0.56 | 0.219 | 0.316 |

Signal features help most on Extracellular / membrane / peroxisome-linked decisions (Δ macro F1 **+0.013**). Exports use the `with_signal` variant.

## B3 — Aux exports

fp16 npz under `runs/aux/` (decoder evidence; Phase A wires these in):

| file | n | shape notes |
|------|--:|-------------|
| `signal_{train,tv,test,fr24p}.npz` | 112983 / 7566 / 8630 / 11231 | `sp_prob[6]`, `cleavage`, `summary[9]` |
| `loc_{train,tv,test,fr24p}.npz` | same | `loc_prob[10]` |

Mean P(SP present): train 0.137, tv 0.131, test 0.153, fr24p 0.156. Mean loc labels at 0.5: ≈1.04–1.09.

Kingdom for the signal start mask is taken from NCBI superkingdom (Eukaryota→`EUKARYA`, Archaea→`ARCHAEA`, Bacteria/unknown→all starts allowed). DeepLoc-train accessions are predicted out-of-fold; all other TraceGO proteins ensemble all fold heads.

## B4 — TV loc→CC AUPR (TV only)

Map each DeepLoc location to its CC GO term and score loc probability vs `gt_tv.tsv`. **Not** scored on test or fresh24.

| location | GO | n_pos | prev | AUPR | AUPR−prev |
|----------|-----|------:|-----:|-----:|----------:|
| Cytoplasm | GO:0005737 | 2703 | 0.357 | 0.438 | +0.081 |
| Nucleus | GO:0005634 | 1779 | 0.235 | 0.459 | +0.224 |
| Extracellular | GO:0005576 | 453 | 0.060 | 0.220 | +0.160 |
| Cell membrane | GO:0005886 | 932 | 0.123 | 0.349 | +0.225 |
| Mitochondrion | GO:0005739 | 450 | 0.059 | 0.405 | +0.346 |
| Plastid | GO:0009536 | 92 | 0.012 | 0.353 | +0.341 |
| Endoplasmic reticulum | GO:0005783 | 265 | 0.035 | 0.211 | +0.176 |
| Lysosome/Vacuole | GO:0005773 | 172 | 0.023 | 0.080 | +0.057 |
| Golgi apparatus | GO:0005794 | 267 | 0.035 | 0.205 | +0.170 |
| Peroxisome | GO:0005777 | 47 | 0.006 | 0.115 | +0.109 |

DeepLoc-train overlap among TV: **985 / 7566**. Held-out (non-overlap) AUPRs remain above prevalence for the major compartments (see `runs/aux/b4_tv_cc_aupr.json`). This is a descriptive check that loc probs track CC GO — not a claim that the loc head alone matches TraceGO CC Fmax.

## Disk / GPU

- Outputs under `runs/{signal,loc,aux}` ≈ **70 MB** total (models + preds + aux npz); raw DTU/DeepLoc tables ≈ 24 MB under `runs/aux/raw/`.
- ESM residue states for DeepLoc held in RAM only (~13 GB tokens); TraceGO train/TV/test/fr24p streamed in batches (no residue cache on disk).
- `CUDA_VISIBLE_DEVICES=0` only; Phase A GPU 1 left untouched.
