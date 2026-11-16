"""True-path propagation and a train-only term filter.

The smoke graph is a stand-in for go-basic `is_a` links. A child term
implies every ancestor. Terms that cover almost every training protein
are dropped, using training records only.
"""

from tracego.data.smoke import SmokeRecord

# Child -> parent. The root stands in for a family label such as GO:0004672.
TOY_PARENTS: dict[str, tuple[str, ...]] = {
    "MF:kinase": ("MF:activity",),
    "MF:atp": ("MF:activity",),
    "MF:binding": ("MF:activity",),
    "MF:activity": ("root:all",),
    "BP:phosphorylation": ("BP:process",),
    "BP:signaling": ("BP:process",),
    "BP:process": ("root:all",),
    "CC:cytoplasm": ("CC:component",),
    "CC:extracellular": ("CC:component",),
    "CC:component": ("root:all",),
}


def ancestors(term: str, parents: dict[str, tuple[str, ...]] | None = None) -> set[str]:
    graph = TOY_PARENTS if parents is None else parents
    seen = {term}
    stack = [term]
    while stack:
        current = stack.pop()
        for parent in graph.get(current, ()):
            if parent not in seen:
                seen.add(parent)
                stack.append(parent)
    return seen


def propagate(terms: tuple[str, ...] | set[str]) -> set[str]:
    closed: set[str] = set()
    for term in terms:
        closed |= ancestors(term)
    return closed


def propagate_records(records: list[SmokeRecord]) -> list[SmokeRecord]:
    print(f"[ontology] parent links={len(TOY_PARENTS)}")
    updated: list[SmokeRecord] = []
    for record in records:
        closed = propagate(record.go_terms)
        added = sorted(closed.difference(record.go_terms))
        print(
            f"[ontology] {record.accession} direct={','.join(record.go_terms)} "
            f"added={','.join(added) if added else '(none)'}"
        )
        updated.append(
            SmokeRecord(
                record.accession,
                record.sequence,
                tuple(sorted(closed)),
                record.synthetic,
            )
        )
    return updated


def filter_terms(
    records: list[SmokeRecord],
    train_index: list[int],
    max_frequency: float,
    min_count: int,
) -> tuple[list[str], list[str]]:
    """Drop near-constant and too-rare terms. Frequencies use the train split."""
    print(f"[ontology] train records={[records[i].accession for i in train_index]}")
    print(
        f"[ontology] held out of the filter="
        f"{[records[i].accession for i in range(len(records)) if i not in set(train_index)]}"
    )
    counts: dict[str, int] = {}
    for index in train_index:
        for term in records[index].go_terms:
            counts[term] = counts.get(term, 0) + 1
    n_train = max(len(train_index), 1)
    kept: list[str] = []
    dropped: list[str] = []
    for term in sorted(counts):
        count = counts[term]
        frequency = count / n_train
        if frequency >= max_frequency:
            decision = "drop near-constant"
            dropped.append(term)
        elif count < min_count:
            decision = "drop rare"
            dropped.append(term)
        else:
            decision = "keep"
            kept.append(term)
        print(
            f"[ontology] {term} train_count={count} "
            f"train_frequency={frequency:.2f} decision={decision}"
        )
    print(f"[ontology] kept={kept}")
    print(f"[ontology] dropped={dropped}")
    return kept, dropped
