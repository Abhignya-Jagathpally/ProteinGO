"""UniProt download for one Pfam family and one broad Gene Ontology term."""

import json
import re
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def _session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def fetch_proteins(config: dict, work: Path) -> pd.DataFrame:
    work.mkdir(parents=True, exist_ok=True)
    cache = work / "uniprot_raw.tsv"
    query = (
        f"(xref:pfam-{config['domain']}) AND "
        f"(go:{config['common_go'].replace('GO:', '')})"
    )
    print(f"[uniprot] query={query}")
    if cache.exists():
        raw = pd.read_csv(cache, sep="\t", dtype=str).fillna("")
        print(f"[uniprot] cache hit rows={len(raw)} path={cache}")
    else:
        session = _session()
        params = {
            "query": query,
            "format": "tsv",
            "fields": "accession,sequence,go_id,reviewed,length",
            "size": 500,
        }
        frames = []
        valid = set()
        next_url = "https://rest.uniprot.org/uniprotkb/search"
        first = True
        pages = 0
        n_rows = 0
        while next_url and n_rows < int(config["max_fetch_rows"]):
            response = session.get(
                next_url,
                params=params if first else None,
                timeout=(20, 120),
            )
            response.raise_for_status()
            if first:
                print(
                    "[uniprot] reported hits="
                    f"{response.headers.get('x-total-results', 'unknown')}"
                )
            page = pd.read_csv(StringIO(response.text), sep="\t", dtype=str).fillna("")
            if page.empty:
                break
            frames.append(page)
            n_rows += len(page)
            pages += 1
            for sequence, go_text in zip(page.iloc[:, 1], page.iloc[:, 2]):
                cleaned = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "X", str(sequence).upper())
                if int(config["min_aa"]) <= len(cleaned) <= int(config["max_aa"]):
                    if re.search(r"GO:\d{7}", str(go_text)):
                        valid.add(cleaned)
            if pages == 1 or pages % 5 == 0:
                print(
                    f"[uniprot] page={pages:03d} fetched={n_rows} "
                    f"valid_candidates={len(valid)}",
                    flush=True,
                )
            match = re.search(r'<([^>]+)>;\s*rel="next"', response.headers.get("Link", ""))
            next_url = match.group(1) if match else None
            first = False
            if len(valid) >= int(config["target_records"]):
                break
        if not frames:
            raise RuntimeError("UniProt returned no rows")
        raw = pd.concat(frames, ignore_index=True)
        raw.to_csv(cache, sep="\t", index=False)
        (work / "uniprot_request.json").write_text(
            json.dumps(
                {
                    "query": query,
                    "rows": len(raw),
                    "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            )
        )
        print(f"[uniprot] wrote {cache} pages={pages} rows={len(raw)}")

    frame = pd.DataFrame(
        {
            "accession": raw.iloc[:, 0].astype(str),
            "sequence": raw.iloc[:, 1].astype(str).str.upper(),
            "go_raw": raw.iloc[:, 2].astype(str),
            "reviewed": raw.iloc[:, 3].astype(str),
            "length_reported": pd.to_numeric(raw.iloc[:, 4], errors="coerce"),
        }
    )
    frame["sequence"] = frame["sequence"].str.replace(
        r"[^ACDEFGHIKLMNPQRSTVWY]", "X", regex=True
    )
    frame["seq_length"] = frame["sequence"].str.len()
    frame["go_direct"] = frame["go_raw"].map(lambda text: set(re.findall(r"GO:\d{7}", text)))
    before = len(frame)
    frame = frame[frame.seq_length.between(int(config["min_aa"]), int(config["max_aa"]))]
    frame = frame[frame["go_direct"].map(bool)]
    frame = frame.drop_duplicates(subset="sequence").reset_index(drop=True)
    print(
        f"[uniprot] raw={before} after length, labels, exact-sequence dedup={len(frame)}"
    )
    print(f"[uniprot] reviewed={frame.reviewed.value_counts().to_dict()}")
    if len(frame) < int(config["min_records"]):
        raise RuntimeError(
            f"only {len(frame)} proteins passed filters, need {config['min_records']}"
        )
    if len(frame) > int(config["target_records"]):
        frame = frame.sample(
            n=int(config["target_records"]), random_state=int(config["seed"])
        ).reset_index(drop=True)
    print(
        f"[uniprot] kept={len(frame)} domain={config['domain']} "
        f"common_go={config['common_go']}"
    )
    return frame
