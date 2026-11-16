"""GO-definition text embeddings from the decoder's own text encoder (flan-t5-small, mean-pooled).

Only the ontology text is used ("name: definition" from go-basic 2023-01-01); no protein record text, so the
embeddings cannot carry a protein's own label names (the ProtGO leakage mode).
"""
from __future__ import annotations

import time

import numpy as np
import torch

from tracego import paths

T5 = "google/flan-t5-small"


def definition_texts(vocab) -> list[str]:
    asp = {"F": "molecular function", "P": "biological process", "C": "cellular component"}
    return [f"{asp[a]}: {n}. {d}" for a, n, d in zip(vocab.aspect, vocab.name, vocab.definition)]


def build(vocab, out=paths.CACHE / "go_def_t5.npy", device="cuda") -> np.ndarray:
    if out.exists():
        E = np.load(out)
        print(f"[godefs] cache hit {E.shape} {out}")
        return E
    from transformers import AutoTokenizer, T5EncoderModel
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(T5)
    enc = T5EncoderModel.from_pretrained(T5).to(device).eval()
    texts = definition_texts(vocab)
    print(f"[godefs] {len(texts):,} definitions; example: {texts[0][:160]}")
    E = np.zeros((len(texts), enc.config.d_model), np.float32)
    with torch.no_grad():
        for s in range(0, len(texts), 256):
            b = tok(texts[s:s + 256], return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            h = enc(**b).last_hidden_state
            m = b["attention_mask"][..., None].float()
            E[s:s + 256] = ((h * m).sum(1) / m.sum(1)).cpu().numpy()
    np.save(out, E)
    print(f"[godefs] embeddings {E.shape} norm mean {np.linalg.norm(E, axis=1).mean():.2f}; {time.time() - t0:.0f}s")
    return E


def latent_init(E: np.ndarray, aspect: np.ndarray, n_latents: int, seed: int = 0) -> np.ndarray:
    """k-means centroids of the definition embeddings, split across namespaces in proportion to sqrt(#terms)."""
    from sklearn.cluster import KMeans
    asps = ["F", "P", "C"]
    w = np.array([np.sqrt((aspect == a).sum()) for a in asps])
    n = np.maximum(1, np.round(n_latents * w / w.sum()).astype(int))
    n[np.argmax(n)] += n_latents - n.sum()
    C = []
    for a, k in zip(asps, n):
        X = E[aspect == a]
        C.append(KMeans(k, n_init=4, random_state=seed).fit(X).cluster_centers_)
    C = np.concatenate(C).astype(np.float32)
    print(f"[godefs] latent init: {dict(zip(asps, n.tolist()))} centroids, shape {C.shape}")
    return C
