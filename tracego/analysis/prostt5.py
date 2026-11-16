"""Phase C2: stream ProstT5_fp16 encoder weights into RAM (no weight cache on disk) and embed sequences.

Public HF repo; no token. Keeps only the T5 encoder (~2.4 GB VRAM in fp16). Optional 3Di CNN from the
ProstT5 github (tiny). Driver: `scripts/analysis_c2_structure.py`.
"""
from __future__ import annotations

import io
from pathlib import Path
import re
from collections import OrderedDict

import numpy as np
import requests
import torch
from transformers import T5EncoderModel, T5Tokenizer

from tracego.analysis.common import log

TAG = "prostt5"
REPO = "Rostlab/ProstT5_fp16"
WEIGHTS = f"https://huggingface.co/{REPO}/resolve/main/pytorch_model.bin"
CFG = f"https://huggingface.co/{REPO}/resolve/main/config.json"
SPIECE = f"https://huggingface.co/{REPO}/resolve/main/spiece.model"
TOK_CFG = f"https://huggingface.co/{REPO}/resolve/main/tokenizer_config.json"
SPECIAL = f"https://huggingface.co/{REPO}/resolve/main/special_tokens_map.json"


def _get(url: str) -> bytes:
    import hashlib
    shm = Path("/dev/shm") / ("prostt5_" + hashlib.md5(url.encode()).hexdigest()[:12] + ".bin")
    if shm.exists() and shm.stat().st_size > 1e9:
        log(TAG, f"RAM-disk cache hit {shm} ({shm.stat().st_size/1e9:.2f} GB)")
        return shm.read_bytes()
    r = requests.get(url, timeout=120, stream=True)
    r.raise_for_status()
    buf = io.BytesIO()
    n = 0
    for chunk in r.iter_content(1 << 20):
        buf.write(chunk)
        n += len(chunk)
        if n // (50 << 20) != (n - len(chunk)) // (50 << 20):
            log(TAG, f"downloaded {n / 1e9:.2f} GB of {url.rsplit('/', 1)[-1]}")
    data = buf.getvalue()
    # best-effort cache on tmpfs (not counted against project disk)
    try:
        if "pytorch_model" in url and Path("/dev/shm").exists():
            shm.write_bytes(data)
            log(TAG, f"cached weights on /dev/shm ({len(data)/1e9:.2f} GB)")
    except Exception as e:
        log(TAG, f"/dev/shm cache skipped: {type(e).__name__}")
    return data


def load_encoder(device="cuda", tmp_tok_dir=None):
    """Stream fp16 weights into RAM, keep encoder.* + shared, build T5EncoderModel. Tokenizer files (~MB) may
    land under tmp_tok_dir (defaults to runs/analysis/c2/tok)."""
    from pathlib import Path
    from tracego.analysis.common import out_dir
    tok_dir = Path(tmp_tok_dir) if tmp_tok_dir else out_dir("c2") / "tok"
    tok_dir.mkdir(parents=True, exist_ok=True)
    for name, url in (("config.json", CFG), ("spiece.model", SPIECE),
                      ("tokenizer_config.json", TOK_CFG), ("special_tokens_map.json", SPECIAL)):
        p = tok_dir / name
        if not p.exists():
            p.write_bytes(_get(url))
            log(TAG, f"wrote {p} ({p.stat().st_size} B)")
    log(TAG, "streaming ProstT5_fp16 pytorch_model.bin into RAM (not written to disk) ...")
    raw = _get(WEIGHTS)
    log(TAG, f"weights in RAM {len(raw) / 1e9:.2f} GB; torch.load ...")
    blob = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    del raw
    # keep encoder + shared
    keep = OrderedDict()
    for k, v in blob.items():
        if k.startswith("encoder.") or k.startswith("shared."):
            keep[k] = v
    del blob
    log(TAG, f"kept {len(keep)} encoder/shared tensors; "
             f"{sum(t.numel() * t.element_size() for t in keep.values()) / 1e9:.2f} GB")
    # Build from config only (no weight files on disk), then load the streamed encoder tensors.
    from transformers import T5Config
    cfg = T5Config.from_pretrained(tok_dir)
    model = T5EncoderModel(cfg)
    # HF T5EncoderModel expects keys without a leading "encoder." for the stack in some versions;
    # try both the streamed names and stripped names.
    missing, unexpected = model.load_state_dict(keep, strict=False)
    if len(missing) > len(keep) // 2:
        keep2 = OrderedDict((k[len("encoder."):] if k.startswith("encoder.") else k, v) for k, v in keep.items())
        # shared.weight -> embed_tokens / shared
        remapped = OrderedDict()
        for k, v in keep2.items():
            if k == "shared.weight":
                remapped["shared.weight"] = v
                remapped["encoder.embed_tokens.weight"] = v
            elif not k.startswith("shared"):
                remapped["encoder." + k if not k.startswith("encoder.") else k] = v
            else:
                remapped[k] = v
        missing, unexpected = model.load_state_dict(remapped, strict=False)
        del keep2, remapped
    log(TAG, f"load_state_dict missing {len(missing)} unexpected {len(unexpected)}")
    if missing[:5]:
        log(TAG, f"  missing sample: {missing[:5]}")
    del keep
    model = model.half().to(device).eval()
    tok = T5Tokenizer.from_pretrained(tok_dir)
    return model, tok


def prep_seq(seq: str) -> str:
    """ProstT5 AA→fold input: uppercase, rare→X, space-separated, <AA2fold> prefix."""
    s = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "X", seq.upper())
    return "<AA2fold> " + " ".join(s)


@torch.no_grad()
def embed_seqs(model, tok, seqs, device="cuda", max_len: int = 1022, batch: int = 4) -> np.ndarray:
    """Mean-pooled encoder embeddings [n, d] fp16."""
    outs = []
    for s in range(0, len(seqs), batch):
        texts = [prep_seq(x[:max_len]) for x in seqs[s:s + batch]]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len + 2).to(device)
        h = model(**enc).last_hidden_state.float()
        m = enc["attention_mask"][..., None].float()
        # drop the prefix token (position 0)
        m[:, 0] = 0
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        outs.append(pooled.half().cpu().numpy())
        if (s // batch) % 20 == 0:
            log(TAG, f"embedded {min(s + batch, len(seqs)):,}/{len(seqs):,}")
    return np.concatenate(outs, 0)
