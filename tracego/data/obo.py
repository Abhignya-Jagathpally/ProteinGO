"""Parse go-basic.obo and close annotations under is_a and part_of."""

import hashlib
from functools import lru_cache
from pathlib import Path

import requests


def download_obo(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 1_000_000:
        print(f"[obo] cache hit bytes={path.stat().st_size} path={path}")
        return path
    url = "https://purl.obolibrary.org/obo/go/go-basic.obo"
    print(f"[obo] downloading {url}")
    response = requests.get(url, timeout=(20, 180))
    response.raise_for_status()
    path.write_bytes(response.content)
    print(f"[obo] wrote bytes={path.stat().st_size}")
    return path


def parse_obo(path: Path):
    terms = {}
    alt_map = {}
    current = None

    def save(item):
        if not item or not item.get("id") or item.get("obsolete"):
            return
        terms[item["id"]] = item
        for alt in item["alt_id"]:
            alt_map[alt] = item["id"]

    for raw in path.open(encoding="utf-8"):
        line = raw.strip()
        if line == "[Term]":
            save(current)
            current = {
                "id": None,
                "name": "",
                "namespace": "",
                "parents": set(),
                "alt_id": [],
                "obsolete": False,
            }
            continue
        if line.startswith("["):
            save(current)
            current = None
            continue
        if current is None:
            continue
        if line.startswith("id: "):
            current["id"] = line[4:]
        elif line.startswith("name: "):
            current["name"] = line[6:]
        elif line.startswith("namespace: "):
            current["namespace"] = line[11:]
        elif line.startswith("alt_id: "):
            current["alt_id"].append(line[8:])
        elif line == "is_obsolete: true":
            current["obsolete"] = True
        elif line.startswith("is_a: "):
            current["parents"].add(line.split()[1])
        elif line.startswith("relationship: part_of "):
            current["parents"].add(line.split()[2])
    save(current)
    for record in terms.values():
        record["parents"] = {
            alt_map.get(parent, parent)
            for parent in record["parents"]
            if alt_map.get(parent, parent) in terms
        }
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:20]
    print(f"[obo] terms={len(terms)} alts={len(alt_map)} sha256={digest}")
    return terms, alt_map


def ancestor_fn(terms):
    @lru_cache(maxsize=None)
    def ancestors(go_id: str):
        if go_id not in terms:
            return frozenset()
        result = {go_id}
        for parent in terms[go_id]["parents"]:
            result.update(ancestors(parent))
        return frozenset(result)

    return ancestors


def close_annotations(direct, alt_map, ancestors) -> set[str]:
    closed = set()
    for term in direct:
        closed.update(ancestors(alt_map.get(term, term)))
    return closed
