"""Tiny labeled sequences for the smoke forward pass.

These records are synthetic. They exist so the pipeline can print shapes
without a UniProt download. They are not a kinase benchmark.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SmokeRecord:
    accession: str
    sequence: str
    go_terms: tuple[str, ...]
    synthetic: bool = True


# Four short sequences and four molecular-function-like labels.
# Labels are toy identifiers, not a claim about these peptides.
_SMOKE = (
    SmokeRecord(
        "SMOKE0001",
        "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL",
        ("MF:kinase", "MF:atp", "CC:cytoplasm"),
    ),
    SmokeRecord(
        "SMOKE0002",
        "MEEKVRFYIQNHGDAYTILPKTLKELGDIVKLTKPGAALLLNAVYGIDVTEIIAKNLEKIKPQVLLGSNGGTVFSAKALQEELGWTAPIVTNDGHTPTAEEIKAYLEAGCKVVLVGDLSKPGTEGDFIGLKEIKPGQVLTFNGGTVFSVKALQERLGWTAP",
        ("MF:kinase", "BP:phosphorylation"),
    ),
    SmokeRecord(
        "SMOKE0003",
        "MGSSHHHHHHSSGLVPRGSHMKFLVLLAIIAVAATVASAAPAPAPAPEEKKDEGDEVQ",
        ("CC:extracellular", "MF:binding"),
    ),
    SmokeRecord(
        "SMOKE0004",
        "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN",
        ("CC:extracellular", "MF:binding", "BP:signaling"),
    ),
)


def load_smoke_records(n: int = 4) -> list[SmokeRecord]:
    records = list(_SMOKE[:n])
    print(f"[data] smoke records: {len(records)} synthetic sequences")
    for record in records:
        print(
            f"[data] {record.accession} length={len(record.sequence)} "
            f"terms={','.join(record.go_terms)} synthetic={record.synthetic}"
        )
    return records
