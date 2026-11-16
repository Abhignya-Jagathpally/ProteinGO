"""Filesystem locations shared by every TraceGO stage."""
from __future__ import annotations

import sys
from pathlib import Path

PFA = Path(__file__).resolve().parents[2]
TRACEGO = PFA / "tracego"
CAFA6 = PFA / "cafa6"
TV2 = CAFA6 / "data" / "proc_gogpt_tv2"
PROC = CAFA6 / "data" / "proc_gogpt"
FRESH = CAFA6 / "data" / "fresh24"
REF = CAFA6 / "runs" / "gogpt_cafa5" / "ref"
OBO = REF / "go-basic_gogpt.obo"
IA = REF / "IA.txt"
RUNS = TRACEGO / "runs"
CACHE = RUNS / "cache"
BUNDLE = CACHE / "bundle"
TAXDUMP = CACHE / "taxdump"


def cafa6_import():
    """Makes `import cafa6` resolve to the repo's scorer package."""
    p = str(CAFA6 / "src")
    if p not in sys.path:
        sys.path.insert(0, p)
