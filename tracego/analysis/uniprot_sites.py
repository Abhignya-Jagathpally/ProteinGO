"""Annotated functional residues from UniProt (REST, fields ft_act_site and ft_binding) for TV v2 queries.

A feature is kept only when the current UniProt sequence equals the bundle sequence, so positions index the same
residues the encoder saw. Binding sites carry their ligand (name + ChEBI id), which maps to a ligand-binding MF
term (LIGAND_GO), so a residue trace for, say, "ATP binding" can be checked against the ATP-binding residues.

  python -m tracego.analysis.uniprot_sites          # -> runs/analysis/c1/tv_sites.json
"""
from __future__ import annotations

import json
import time
from collections import Counter

import numpy as np
import requests

from tracego.analysis.common import log, out_dir

TAG = "sites"
URL = "https://rest.uniprot.org/uniprotkb/accessions"

# UniProt ligand name -> ligand-binding MF term (go-basic names checked against the vocabulary at load time)
LIGAND_GO = {
    "ATP": "GO:0005524", "ADP": "GO:0043531", "GTP": "GO:0005525", "GDP": "GO:0019003",
    "Zn(2+)": "GO:0008270", "Mg(2+)": "GO:0000287", "Ca(2+)": "GO:0005509", "Mn(2+)": "GO:0030145",
    "Cu(2+)": "GO:0005507", "Cu cation": "GO:0005507", "Fe(2+)": "GO:0005506", "Fe(3+)": "GO:0005506",
    "Fe cation": "GO:0005506", "Ni(2+)": "GO:0016151", "Co(2+)": "GO:0050897", "K(+)": "GO:0030955",
    "Na(+)": "GO:0031402", "heme": "GO:0020037", "heme b": "GO:0020037", "heme c": "GO:0020037",
    "NAD(+)": "GO:0051287", "NADH": "GO:0051287", "NADP(+)": "GO:0050661", "NADPH": "GO:0050661",
    "FAD": "GO:0071949", "FMN": "GO:0010181", "S-adenosyl-L-methionine": "GO:1904047",
    "pyridoxal 5'-phosphate": "GO:0030170", "[4Fe-4S] cluster": "GO:0051539", "[2Fe-2S] cluster": "GO:0051537",
    "thiamine diphosphate": "GO:0030976", "CoA": "GO:0120225", "chlorophyll a": "GO:0016168",
    "Cl(-)": "GO:0031404", "DNA": "GO:0003677", "RNA": "GO:0003723", "substrate": None,
}
CATALYTIC = "GO:0003824"
METAL = "GO:0046872"
METALS = {"Zn(2+)", "Mg(2+)", "Ca(2+)", "Mn(2+)", "Cu(2+)", "Cu cation", "Fe(2+)", "Fe(3+)", "Fe cation", "Ni(2+)",
          "Co(2+)", "K(+)", "Na(+)"}


def fetch(accs, batch: int = 100, sleep_s: float = 0.2) -> dict:
    """{accession: {"seq": str, "features": [(type, start, end, ligand, chebi)]}} from UniProt REST."""
    out = {}
    for s in range(0, len(accs), batch):
        chunk = list(accs[s:s + batch])
        for attempt in range(4):
            try:
                r = requests.get(URL, params={"accessions": ",".join(chunk), "format": "json",
                                              "fields": "accession,sequence,ft_act_site,ft_binding"}, timeout=60)
                r.raise_for_status()
                break
            except requests.RequestException as e:
                log(TAG, f"batch {s} attempt {attempt}: {type(e).__name__}; retrying")
                time.sleep(2 + 3 * attempt)
        else:
            continue
        for rec in r.json().get("results", []):
            feats = []
            for f in rec.get("features", []):
                lo, hi = f["location"]["start"].get("value"), f["location"]["end"].get("value")
                if lo is None or hi is None:
                    continue
                lig = f.get("ligand", {})
                feats.append((f["type"], int(lo), int(hi), lig.get("name", ""), lig.get("id", "")))
            out[rec["primaryAccession"]] = {"seq": rec.get("sequence", {}).get("value", ""), "features": feats}
        if (s // batch) % 10 == 0:
            log(TAG, f"fetched {min(s + batch, len(accs)):,}/{len(accs):,} accessions; with features "
                     f"{sum(bool(v['features']) for v in out.values()):,}")
        time.sleep(sleep_s)
    return out


def site_sets(entry: dict, L: int) -> dict:
    """{label: sorted 0-based residue positions < L}: 'act' (active sites), 'bind' (all binding sites),
    'functional' (union) and one 'lig:<name>' per ligand. Binding ranges longer than 3 residues are dropped
    (those are usually loops whose residues are not individually annotated)."""
    sets = {"act": set(), "bind": set()}
    for typ, lo, hi, lig, _ in entry["features"]:
        if hi - lo > 2:
            continue
        pos = {p - 1 for p in range(lo, hi + 1) if p - 1 < L}
        if typ == "Active site":
            sets["act"] |= pos
        elif typ == "Binding site":
            sets["bind"] |= pos
            if lig:
                sets.setdefault(f"lig:{lig}", set()).update(pos)
    sets["functional"] = sets["act"] | sets["bind"]
    return {k: sorted(v) for k, v in sets.items() if v}


def main():
    from tracego.data.bundle import Bundle
    b = Bundle()
    tv = b.rows("tv")
    accs = b.ids[tv]
    log(TAG, f"TV v2 queries {len(accs):,}; querying UniProt REST for active / binding sites")
    t0 = time.time()
    raw = fetch(accs)
    seq = dict(zip(b.ids[tv], b.prot.seq.values[tv]))
    kept, mism, empty = {}, 0, 0
    lig = Counter()
    for a, e in raw.items():
        if not e["features"]:
            empty += 1
            continue
        if e["seq"] != seq.get(a):
            mism += 1
            continue
        kept[a] = e
        lig.update({f[3] for f in e["features"] if f[3]})
    n_act = sum(any(f[0] == "Active site" for f in e["features"]) for e in kept.values())
    log(TAG, f"done in {time.time() - t0:.0f}s: returned {len(raw):,}; no site features {empty:,}; sequence changed "
             f"{mism:,}; kept {len(kept):,} (with active site {n_act:,})")
    log(TAG, f"top ligands: {lig.most_common(25)}")
    vocab = set(b.terms)
    miss = sorted({g for g in LIGAND_GO.values() if g and g not in vocab})
    log(TAG, f"ligand GO terms not in the vocabulary: {miss}")
    od = out_dir("c1")
    json.dump(kept, open(od / "tv_sites.json", "w"))
    sizes = [len(site_sets(e, len(e["seq"])).get("functional", [])) for e in kept.values()]
    log(TAG, f"functional residues per protein p10/p50/p90 {np.percentile(sizes, [10, 50, 90]).tolist()}; "
             f"wrote {od / 'tv_sites.json'}")


if __name__ == "__main__":
    main()
