# TraceGO

Retrieval-conditioned, hierarchy-constrained instruction decoding of Gene Ontology terms. The design is in `docs/PLAN.md`; the decisions made during development, and the evidence for each, are in section 8 of that file.

Each protein goes through ESM2-35M once. LoRA is applied to layers 9-12 only, and that single pass produces:

- the latents of a GO-grounded Perceiver resampler, used as soft prompts;
- a whitened mean-pool, used to find retrieval candidates;
- layer-8 residue codes, used to re-rank candidates by identity along the alignment the embeddings induce.

No MMseqs2 or DIAMOND output enters the model.

A flan-t5-small instruction model then reads the instruction "list the {aspect} GO terms of this protein. Organism: X. Evidence: ...", followed by the neighbours' GO tokens. It emits GO tokens under the hierarchy constraint (HC-ELO). A commit-conditioned membership head (CCMS) scores every term. That head is anchored to the neighbour vote through a copy bias.

All selection is done on TV v2 / TV-remote. The GO-GPT temporal test and fresh24 FR24-P are scored descriptively, and nothing is written to `cafa6/runs/test_looks.md`.

Phase A (frozen): homology-curriculum decoder (`dec_curriculum`) + hybrid retrieval (exact 8-mer union on TV/test; no k-mer on TV-remote/FR24-P). Official once-only means: **test** TraceGO 0.7129 / retrieval-only 0.7173 (DIAMOND 0.7206, HiGO-R2 0.7178; 100%-id bin TraceGO 0.882 / vote 0.894 vs DIAMOND 0.898); **FR24-P** TraceGO 0.4862 (SPROF 0.5147, HiGO-R2 0.4794). Detail in `runs/phaseA/RESULTS.md`; tables in `runs/final/`.

## Pipeline

Run everything from this directory with `export PYTHONPATH=$PWD` and `../bin/python`. Every stage prints its progress.

```bash
python -m tracego.data.bundle                        # runs/cache/bundle: proteins, labels, ontology, taxa
python -m tracego.train.cache_codes --config configs/full.yaml               # shared layer-8 residue codes
python -m tracego.train.train_encoder --config configs/full.yaml --fold 0    # and --fold 1; builds GO-definition embeddings on first use
python -m tracego.train.build_evidence --config configs/full.yaml --fold 0   # and --fold 1
python -m tracego.train.train_decoder --config configs/full.yaml --tag anchor \
    decoder.epochs=3 decoder.anchor=true decoder.anchor_l2=0.05
python -m tracego.eval.predict --config configs/full.yaml --tag anchor --benches tv,test,fr24p \
    decoder.anchor=true decoder.anchor_l2=0.05
python scripts/knn_baseline.py --config configs/full.yaml --benches tv,test,fr24p     # retrieval-only vote
python -m tracego.train.train_classifier --config configs/full.yaml --tag clf --benches tv,test,fr24p
python scripts/final_eval.py                         # runs/final: summary, per-bin, extended metrics, bootstrap
python scripts/extras.py                             # UMAP / NMI / ARI / silhouette / CKA
python -m tracego.eval.trace --config configs/full.yaml --tag anchor --out runs/full/trace
```

`configs/dev.yaml` runs the same pipeline on a 20k-protein subset.

Ablations use the same commands with these overrides:

- `encoder.tag=frozen35m encoder.lora_r=0`
- `decoder.evidence=false decoder.copy_bias=false`
- `decoder.constrained=false`
- `decoder.def_init=false`
- `decoder.ia_weight=false`

The two scripts `runs/ablate_gpu0.sh` and `runs/ablate_gpu1.sh` run them.

Diagnostics that informed the design are `scripts/baseline_frozen.py` (frozen ESM2 probe and kNN), `scripts/probe_rerank.py`, `scripts/probe_identity.py`, `scripts/probe_aaid.py` and `scripts/probe_calib.py`. They use TV v2 only, and DIAMOND hits only for recall diagnostics.

## Layout

```
configs/            dev.yaml, full.yaml
docs/PLAN.md        design, novelty, literature, departures
tracego/data        bundle.py (labels, ontology, taxa), godefs.py, instructions.py (prompts, collator)
tracego/models      encoder.py (ESM2 + LoRA), resampler.py, adapter.py, retrieval.py, decoder.py
tracego/train       train_encoder.py, cache_codes.py, build_evidence.py, train_decoder.py, train_classifier.py
tracego/eval        score.py (official scorer wrapper), predict.py, trace.py
scripts/            baselines, probes, final_eval.py, extras.py
runs/               weights, caches, predictions (gitignored)
```

# TRACE (earlier work in this directory)

Residue-level Gene Ontology readout on frozen ESM2 states.

The smoke path checks that a mean-pooled vector and a residue-local head can both be trained on the same frozen features. Synthetic sequences are used so the script runs before the kinase download.

The comparison that matches the kinase notebook is:

```bash
python scripts/run_evolve.py
```

It keeps the notebook's 50 percent identity clustering and frequency-ranked Gene Ontology vocabulary. Predictions start from a train-only MMseqs2 homology prior. A small network then corrects that prior using the difference between the frozen ESM2-8M vector of the query and its nearest training homolog, plus the difference in charge, hydrophobicity, and length. Test F1 uses the threshold chosen on validation. Scores land in `runs/evolve/benchmark_scores.csv`.

`python scripts/run_kinase.py` is the earlier 30 percent residue-head run. It does not beat the notebook. `python scripts/run_small.py` is only a synthetic wiring check.
