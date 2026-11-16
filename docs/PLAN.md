# TraceGO plan: retrieval-conditioned, hierarchy-constrained instruction decoding of GO terms

Written before any TraceGO code. Selection data is TV v2 only (`cafa6/data/proc_gogpt_tv2`, 7,566 proteins,
plain and test-like weighted mean IA-Fmax). The GO-GPT temporal test (8,630) and fresh24 FR24-P (11,231) are
scored once per frozen configuration, descriptively, with `--no-log` semantics: nothing is appended to
`cafa6/runs/test_looks.md` and nothing is selected on them.

## 1. Pipeline

```
sequence ──► ESM2-35M + LoRA(q,v; r=16)  ──► residue states H (L×480)        [encoded ONCE, cached]
                     │                         │
                     │                         ├─► GO-grounded Perceiver resampler: 16 latent queries,
                     │                         │   initialised from k-means centroids of GO-definition
                     │                         │   text embeddings  ──► 16 soft-prompt latents Z (16×512)
                     │                         ├─► whitened mean-pool h (candidate space)
                     │                         └─► layer-8 residue codes (PCA-64, shared by all folds)
                     ▼
     cross-fitted retrieval in the encoder's space (no MMseqs2 / DIAMOND): cosine top-100 on h, re-ranked by
     identity along the residue-code alignment (see section 8); top-k training proteins with their
     identity bin + taxonomic LCA rank (NCBI taxdump) + their GO leaves
                     ▼
 flan-t5-small instruction model (text-to-text):
   encoder input = [Z soft prompt] + "Instruction: list the molecular function GO terms of this protein.
                   Organism: <name>. Evidence: <nb> <sim_17> <tax_genus> GO:0004672 GO:0005524 <nb> ..."
   decoder output = GO tokens (one new token per vocabulary term, embedding initialised from its OBO
                   name + definition through the T5 encoder), hierarchy-constrained, then </s>
                     ▼
 commit-conditioned membership head on decoder states ──► continuous per-term scores ──► max-propagation
```

Every protein goes through the adapted encoder exactly once. Retrieval, soft prompts and evidence all come from that
one pass. MMseqs2/DIAMOND are used only for the existing split construction and for the identity bins of the
descriptive breakdown (the `max_pident` already shipped with the repo), never as model input.

## 2. Novel algorithmic pieces

1. **Hierarchy-constrained set decoding with an expected-linearisation objective (HC-ELO).** A GO annotation is a
   set closed under ancestors, not a sequence. At decoding step s with emitted set S_s, the legal next tokens are the
   *frontier* F(S_s) = {t ∉ S_s : every in-vocabulary ancestor of t is in S_s} plus `</s>`. Any prefix of a
   topological order is ancestor-closed, so every output (including a truncated one) is true-path consistent by
   construction. Training feeds one random topological linearisation per example (order-invariant teacher forcing)
   but the loss at each step is the cross-entropy against the *uniform distribution over all available true terms*
   A_s = Y ∩ F(S_s), with the softmax restricted to F(S_s) ∪ {</s>}. This equals the expectation of the next-token
   loss over every topological linearisation that extends the prefix, so the gradient no longer depends on the
   sampled order (lower variance than OAXE-style single-order or Hungarian matching), and the model never spends
   probability on illegal tokens.
2. **Commit-conditioned membership scoring (CCMS).** Fmax needs a continuous score for every term; GO-GPT obtains it
   by counting ten samples. A sigmoid membership head on each decoder state predicts which terms of the remaining set
   Y \ S_s belong to the protein, trained with IA-weighted BCE at randomly chosen prefix positions. At inference one
   greedy constrained decode is run; a committed term gets the membership probability from the step at which it
   was committed (conditioned on its committed ancestors), an uncommitted term gets the probability at the final
   state. One decode replaces ten samples, scores are calibrated by BCE, and commits condition later scores
   (a hierarchy-ordered classifier chain).
3. **Cross-fitted retrieval in the adapted space.** A supervised adapter memorises its training proteins' labels, so
   train-to-train neighbours look more label-consistent than test-to-train neighbours (stacking leakage). The
   adapter is trained twice on two halves of the training proteins; a training query's soft prompt and evidence
   come from the adapter that did not see it, with references embedded by the same adapter. At test time the
   decoder runs once in each fold's space and the two score matrices are averaged. This gives the decoder evidence
   with the same reliability it will see on new proteins.
4. **Copy bias (pointer-generator) from the evidence.** The membership logit of term t adds a learned monotone
   function of the similarity-weighted neighbour vote for t (See et al. 2017), so rare and deep terms that the
   decoder has few examples of can still be copied from a close homologue.
5. **IA-weighted losses** (Clark and Radivojac 2013): BCE and set-CE are weighted per term by (IA_t + 1), so deep,
   sparse terms that dominate IA-weighted Fmax are not drowned by roots.

## 3. Literature grounding

- Instruction / text-to-text protein annotation: GO-GPT and BioReason-Pro (Fallahpour et al. 2026; ten-sample
  frequency baseline), Prot2Text / Prot2Text-V2 (Abdine et al. AAAI 2024; layout followed here), ProtNLM, ProtST.
- Soft-prompt bridges: Perceiver / Perceiver-IO (Jaegle et al. 2021), Flamingo resampler (Alayrac et al. 2022),
  BLIP-2 Q-Former (Li et al. 2023).
- PEFT: LoRA (Hu et al. ICLR 2022), its use on ESM2 (Schmirler et al. 2024, Nat. Commun.).
- Retrieval: REALM (Guu et al. 2020), RETRO (Borgeaud et al. 2022), kNN-LM (Khandelwal et al. 2020);
  embedding-based annotation transfer goPredSim (Littmann et al. 2021), PLMSearch (Liu et al. 2024).
- Hierarchy: C-HMCNN max-constraint (Giunchiglia and Lukasiewicz 2020), true-path rule, constrained decoding
  (Hokamp and Liu 2017; Anderson et al. 2017).
- Set generation: Order Matters (Vinyals et al. 2016), Seq2Set (Yang et al. 2019), OAXE (Du et al. 2021),
  classifier chains (Read et al. 2011).
- Label leakage through term names: ProtGO; here definition text is only taken from the OBO, never from protein records.
- Cross-fitting: double machine learning (Chernozhukov et al. 2018), stacked generalisation (Wolpert 1992).

## 4. Data

- Training proteins: `proc_gogpt_tv2` split == train (112,983), labels = T0-stripped propagated labels over the
  7,901-term vocabulary (`Y.npz`, train rows only; test rows of that file are never read).
- Selection: TV v2 queries (7,566) against v2 training references with T0 labels.
- Test / fresh24: the same frozen model, references = GO-GPT train + val with full labels (the repo's
  `knn_trainval` reference set, 133,054), exactly as HiGO-R2 does.
- Development: a 20k training-protein subset (seeded), then the full set.
- GO-GPT validation (random split) only to monitor convergence.

## 5. Experiments

1. Raw ESM2 needs processing: frozen ESM2-35M mean-pool + linear probe, and frozen-embedding kNN, versus the
   adapted model, same split (TV v2).
2. Main: TraceGO at 20k, then full scale.
3. Ablations (TV v2 for decisions, test descriptive): frozen encoder vs LoRA; no retrieval evidence (no evidence
   tokens, no copy bias); no hierarchy constraint (free-order, unmasked decoding); no definition-text init (random
   GO token embeddings and latent queries); no IA weighting; classifier only (membership at step 0, no decoding)
   and a separate MLP classifier on the same cached features without the T5.
4. Metrics: official IA-weighted Fmax per namespace and mean (`knn_sharp.Scorer`), unweighted Fmax, IA S-min,
   term-centric AUPR / AUC, protein-centric MCC at the Fmax threshold; sparse terms (train prevalence < 20 and
   20-100); identity bins of the descriptive breakdown; paired bootstrap CIs (1,000 reps) against the best baseline
   on each benchmark.
5. Extras if time allows: UMAP + NMI/ARI/silhouette of embeddings before/after the adapter, linear CKA frozen vs
   LoRA, ESM contact maps for a few traced examples. Deferred: MD, signal peptides, structure contrastive learning.

## 6. Reverse trace

For each emitted GO term: (i) decoder cross-attention to the 16 soft-prompt latents, composed with the resampler's
latent-to-residue attention (attention rollout) into a residue profile; (ii) evidence attribution (which neighbour
tokens the term attended to); (iii) GO→sequence retrieval: training proteins whose retrieval embedding is closest to
the term's output embedding mapped into r-space, and the neighbours that carried the term.

## 7. Risks stated up front

The bar is a DIAMOND kNN whose margin comes from >= 95% identity proteins (52% of the test). Embedding retrieval
must match DIAMOND's ranking among near-identical sequences; if it cannot, TraceGO will lose in those bins whatever
the decoder does. This is measured first (frozen and adapted embedding kNN against the DIAMOND kNN on TV v2) and
reported plainly.

## 8. Departures made during development (all decided on TV v2)

| change | reason (TV v2 test-like IA-Fmax of the neighbour vote; DIAMOND kNN = 0.7168) |
|---|---|
| contrastive retrieval head r dropped (`ret_weight 0`) | r-space kNN lost to whitened mean-pooled states (dev) |
| candidates = cosine top-100 on whitened adapted mean-pool h | frozen mean-pool 0.658, whitened 0.667 |
| re-rank by late interaction on layer-8 residue codes (PCA-64, fp16) | MaxSim max(q2c, c2q) 0.7064 |
| LoRA on layers 9-12 only (`lora_from_layer 8`) | layer-8 states are then identical in every fold and in the frozen ablation, so one residue-code cache serves all variants (checked bit-identical) |
| re-rank score = chain identity (residues matched above cos 0.9 on a diagonal run) | 0.7143 |
| ... with the matched amino acids required to agree (`chainaa0.0`) | 0.7173 (mean 0.7251); identity read along the alignment the embeddings induce, no second search |
| vote sharpening T = 0.05 in identity units, k = 50 | grid on TV |
| decoder training queries resampled to the TV top-1 evidence-score distribution (`shift_weight`) | training proteins are redundancy-reduced (top-1 score p50 0.63 vs 0.97 on TV), so without it the decoder learns to distrust close homologues and falls below the vote |
| best decoder epoch kept by TV step0 test-like | decoder overfits training queries after epoch 0-1 |
| membership anchored to the vote (`anchor`: copy weight frozen at 1, residual L2 0.05) | unanchored, the learned copy weight fell from 1 to ~0.33 and TV dropped 0.007-0.015 below the vote in every namespace; anchored, the decoder sits on the vote (0.7044 vs 0.7050 vocab-term test-like) |

Negative results kept out of the model: a learned pair-feature calibrator (0.7136), a windowed residue-code candidate
channel for fragment queries (+0.001), layer 12 vs 8 and 64 vs 128 code dims (equal), length-weighted MaxSim.
The residual gap to DIAMOND sits in the 100%-identity bin, where 7.5% of DIAMOND's best hits (mostly fragments,
median length ratio 0.25) are not in the pooled top-100 candidates.

Phase A (TV / TV-remote selection; test and fresh24 scored once at the end):

| change | reason |
|---|---|
| TV-remote mask (`runs/phaseA/tv_remote.npz`) | FR24-P-matched identity mix on TV v2; selection target for fresh24-oriented changes. Baseline anchored TraceGO mix mean 0.5387 (vote 0.5376) |
| Exact 8-mer candidate channel + `max(chainaa, kcov)` (`retrieval.kmer`, `kmer_comb=max`) | Needed: cosine top-100 misses 7.5% of 100%-id DIAMOND hits; union recalls 100% of them. TV vote mean 0.7357 / 100%-bin 0.7688 (was 0.7250 / 0.7494). Query-normalised re-rank (`chainaaq`) loses on TV |
| k-mer off for TV-remote vote | Under the remote mask the k-mer channel drops TV-remote vote 0.5376 → 0.5233 (near-identical refs are hidden, so coverage hits add noise) |
| Frozen large-encoder pools (ESM2-650M, ESM-C 600M, ProtT5-XL) | Cosine-only kNN: ESM-C best (TV 0.700 / TV-remote 0.530) but all trail LoRA-35M + chainaa re-rank (0.725); no LoRA/QLoRA adaptation of the large models yet (disk) |
| Homology-curriculum decoder (`train_cur` evidence, `decoder.homology_token`) | Hide high-identity refs at train time (FR24-P cut mix + 25% unmasked); expose `<hom_*>` + best-sim tokens. **A1 go passed**: vs `_cur` (k-mer) vote on TV-remote mix +0.0049 [0.0029, 0.0067]; vs non-k-mer TV-remote vote +0.0020 [0.0006, 0.0038]; plain TV not worse than the k-mer vote (CI includes 0) and +0.0097 [0.0074, 0.0119] vs prior `dec_anchor`. Frozen decode: k-mer on TV/test, no k-mer on TV-remote/FR24-P. Once-only: test 0.7129 (pre-A 0.7043; DIAMOND 0.7206); FR24-P 0.4862 (pre-A 0.4854; SPROF 0.5147). See `runs/phaseA/RESULTS.md` |
