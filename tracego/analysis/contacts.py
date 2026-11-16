"""Phase C3: ESM2-35M contact-map confidence vs per-protein GO F and homolog count.

Uses the pretrained (not LoRA) contact head — the regression was fit on pretrained attentions
(`EsmForMaskedLM.predict_contacts`, as in `tracego.eval.trace`).

  plot_contact_map(seq, contacts, out_png, profile=None, sites=None, title="")
  contact_confidence(contacts, min_sep=12) -> dict of scalars
"""
from __future__ import annotations

import numpy as np


def contact_confidence(contacts: np.ndarray, min_sep: int = 12) -> dict:
    """Top-L probability mass and mean top-L probability for long-range contacts (|i-j| >= min_sep)."""
    L = contacts.shape[0]
    idx = np.triu_indices(L, k=min_sep)
    vals = contacts[idx]
    if len(vals) == 0:
        return {"topL_mean": 0.0, "topL_mass": 0.0, "L": L, "n_lr": 0}
    k = min(L, len(vals))
    top = np.partition(vals, -k)[-k:]
    return {"topL_mean": float(top.mean()), "topL_mass": float(top.sum() / max(L, 1)),
            "L": L, "n_lr": int(len(vals))}


def plot_contact_map(seq: str, contacts: np.ndarray, out_png, profile=None, sites=None, title: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    L = len(seq)
    C = contacts[:L, :L]
    fig = plt.figure(figsize=(6.5, 7.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.18], hspace=0.05)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(C, cmap="Greys", origin="lower", vmin=0, vmax=max(0.2, float(np.percentile(C, 99))))
    ax.set_title(title or f"ESM2-35M contacts L={L}")
    ax.set_xlabel("residue j")
    ax.set_ylabel("residue i")
    if sites:
        for s in sites:
            if 0 <= s < L:
                ax.axhline(s, color="C1", lw=0.4, alpha=0.7)
                ax.axvline(s, color="C1", lw=0.4, alpha=0.7)
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax)
    if profile is not None:
        p = np.asarray(profile, float)[:L]
        ax2.fill_between(np.arange(L), p / (p.max() + 1e-9), color="C0", alpha=0.7)
        ax2.set_ylabel("trace")
    else:
        ax2.set_axis_off()
    if sites:
        for s in sites:
            if 0 <= s < L:
                ax2.axvline(s, color="C1", lw=0.8)
    ax2.set_xlim(0, L - 1)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
