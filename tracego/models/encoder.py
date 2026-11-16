"""ESM2 encoder with an optional LoRA adapter on the attention query/value projections."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

MAX_RES = 1022


def esm_tokenizer(name: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name)


class ESMEncoder(nn.Module):
    def __init__(self, name: str, lora_r: int = 0, lora_alpha: int = 32, lora_dropout: float = 0.05,
                 targets=("query", "value"), lora_from_layer: int = 0):
        """lora_from_layer = L puts the adapter on transformer layers L.. only (0-based), so hidden_states[L]
        and below stay identical to the pretrained model (shared residue codes for retrieval)."""
        super().__init__()
        from transformers import AutoModel
        esm = AutoModel.from_pretrained(name, add_pooling_layer=False)
        self.hidden = esm.config.hidden_size
        self.lora = lora_r > 0
        if self.lora:
            from peft import LoraConfig, get_peft_model
            n_layers = esm.config.num_hidden_layers
            cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                             target_modules=list(targets), bias="none",
                             layers_to_transform=list(range(lora_from_layer, n_layers)), layers_pattern="layer")
            esm = get_peft_model(esm, cfg)
        else:
            for p in esm.parameters():
                p.requires_grad_(False)
        self.esm = esm

    def forward(self, input_ids, attention_mask, output_attentions=False):
        out = self.esm(input_ids=input_ids, attention_mask=attention_mask, output_attentions=output_attentions)
        return out

    def residue_mask(self, input_ids, attention_mask):
        """True on residue tokens (no <cls>, <eos>, padding)."""
        m = attention_mask.bool().clone()
        m[:, 0] = False
        lens = attention_mask.sum(1)
        m[torch.arange(len(lens)), lens - 1] = False
        return m


def length_batches(lengths: np.ndarray, max_tokens: int, max_batch: int = 512, shuffle: bool = False,
                   rng: np.random.Generator | None = None):
    """Batches of row positions with sum(padded length) <= max_tokens; length-sorted (bucketed when shuffling)."""
    L = np.minimum(lengths, MAX_RES) + 2
    if shuffle:
        noise = rng.uniform(0.9, 1.1, len(L))
        order = np.argsort(L * noise)
    else:
        order = np.argsort(L)
    batches, cur, cur_max = [], [], 0
    for i in order:
        m = max(cur_max, L[i])
        if cur and (m * (len(cur) + 1) > max_tokens or len(cur) >= max_batch):
            batches.append(np.array(cur))
            cur, m = [], L[i]
        cur.append(i)
        cur_max = m
    if cur:
        batches.append(np.array(cur))
    if shuffle:
        rng.shuffle(batches)
    return batches


def tokenize(tok, seqs, device):
    seqs = [s[:MAX_RES] for s in seqs]
    enc = tok(seqs, return_tensors="pt", padding=True)
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)
