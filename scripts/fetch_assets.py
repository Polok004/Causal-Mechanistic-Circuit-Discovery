#!/usr/bin/env python
"""Download everything the pipeline needs, once, so analysis runs stay offline.

    python scripts/fetch_assets.py --all
    python scripts/fetch_assets.py --models --pdb 1A2Y 4HHB
    python scripts/fetch_assets.py --corpus --corpus-size 10000

Keeping every download in one script, rather than fetching lazily inside the
analysis code, is deliberate. An experiment that downloads while it runs is an
experiment whose inputs can change between runs without anything in the code
changing, and whose failure mode on a flaky connection is a half-finished sweep
rather than a clear error. After this script has run once, everything else works
with the network off.

Nothing here is required to run the test suite: the tests construct an
architecture-identical ESM-2 with random weights and use a bundled miniature
structure, precisely so that correctness can be checked without a download.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ASSETS = ROOT / "assets"
MODEL_DIR = ASSETS / "models"
PDB_DIR = ASSETS / "pdb"
DATA_DIR = ASSETS / "data"

RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
UNIPROT_STREAM = (
    "https://rest.uniprot.org/uniprotkb/stream"
    "?query=reviewed:true+AND+length:[{min_len}+TO+{max_len}]"
    "&format=fasta&size={size}"
)
PROTEINGYM_URL = (
    "https://marks.hms.harvard.edu/proteingym/DMS_ProteinGym_substitutions.zip"
)

DEFAULT_MODELS = ["facebook/esm2_t6_8M_UR50D", "facebook/esm2_t12_35M_UR50D"]
DEFAULT_PDBS = ["1A2Y", "4HHB", "1UBQ"]


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n}B"


def fetch_models(model_ids: list[str]) -> None:
    """Pull ESM-2 checkpoints into the local cache.

    Uses ``from_pretrained`` rather than a raw download so the cache layout is
    exactly what ``ESMWrapper.from_pretrained`` will look for later.
    """
    from transformers import AutoTokenizer, EsmForMaskedLM

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for model_id in model_ids:
        print(f"[assets] fetching {model_id} -> {MODEL_DIR}")
        AutoTokenizer.from_pretrained(model_id, cache_dir=str(MODEL_DIR))
        model = EsmForMaskedLM.from_pretrained(model_id, cache_dir=str(MODEL_DIR))
        n_params = sum(p.numel() for p in model.parameters())
        print(
            f"[assets]   ok: {model.config.num_hidden_layers} layers, "
            f"d_model={model.config.hidden_size}, {n_params / 1e6:.1f}M params"
        )
        del model


def fetch_pdb(pdb_ids: list[str]) -> None:
    """Download PDB entries from RCSB."""
    import requests

    PDB_DIR.mkdir(parents=True, exist_ok=True)
    for pdb_id in pdb_ids:
        pdb_id = pdb_id.strip().upper()
        out = PDB_DIR / f"{pdb_id.lower()}.pdb"
        if out.exists():
            print(f"[assets] {out.name} already present, skipping")
            continue
        url = RCSB_URL.format(pdb_id=pdb_id)
        print(f"[assets] fetching {url}")
        resp = requests.get(url, timeout=60)
        if resp.status_code != 200:
            print(f"[assets]   FAILED ({resp.status_code}) — is {pdb_id} a valid PDB id?")
            continue
        out.write_bytes(resp.content)
        print(f"[assets]   wrote {out.name} ({_human(len(resp.content))})")


def fetch_corpus(size: int, min_len: int, max_len: int) -> None:
    """Download a reviewed-SwissProt FASTA subset for SAE training."""
    import requests

    out_dir = DATA_DIR / "corpus"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "sequences.fasta"

    url = UNIPROT_STREAM.format(min_len=min_len, max_len=max_len, size=size)
    print(f"[assets] fetching {size} reviewed sequences ({min_len}-{max_len} aa) from UniProt")
    resp = requests.get(url, timeout=600, stream=True)
    resp.raise_for_status()

    total = 0
    with out.open("wb") as fh:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            fh.write(chunk)
            total += len(chunk)
    n_seqs = sum(1 for line in out.open() if line.startswith(">"))
    print(f"[assets]   wrote {out} ({_human(total)}, {n_seqs} sequences)")

    if n_seqs < size * 0.5:
        print(
            f"[assets]   NOTE: got {n_seqs} sequences for a request of {size}. "
            "UniProt caps streamed results; re-run with a narrower length range "
            "if you need more."
        )


def fetch_proteingym() -> None:
    """Download and unpack the ProteinGym substitution benchmark."""
    import io
    import zipfile

    import requests

    out_dir = DATA_DIR / "ProteinGym"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "DMS_substitutions"
    if target.exists() and any(target.glob("*.csv")):
        print(f"[assets] ProteinGym already unpacked at {target}, skipping")
        return

    print("[assets] fetching ProteinGym substitutions (large, several hundred MB)")
    print(f"[assets]   {PROTEINGYM_URL}")
    resp = requests.get(PROTEINGYM_URL, timeout=3600, stream=True)
    if resp.status_code != 200:
        print(
            f"[assets]   FAILED ({resp.status_code}). ProteinGym release URLs move between "
            "versions; check https://proteingym.org for the current link and pass it with "
            "--proteingym-url."
        )
        return

    buf = io.BytesIO()
    for chunk in resp.iter_content(chunk_size=1 << 20):
        buf.write(chunk)
    buf.seek(0)
    with zipfile.ZipFile(buf) as zf:
        zf.extractall(out_dir)
    n_csv = len(list(out_dir.rglob("*.csv")))
    print(f"[assets]   unpacked {n_csv} assay CSVs under {out_dir}")


def write_clinvar_template() -> None:
    """Write the expected ClinVar table format.

    ClinVar itself is not auto-downloadable in a form that includes protein
    sequences: the variant summary gives HGVS notation, and turning that into
    (wt_sequence, mutation) needs a RefSeq/UniProt join that depends on which
    transcript set you trust. Rather than bake in a choice that would silently
    determine the results, the script documents the target schema and leaves the
    join to you.
    """
    out_dir = DATA_DIR / "clinvar"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "missense_pairs.template.tsv"
    out.write_text(
        "gene\tprotein_id\tpdb_id\twt_sequence\tmutant\tclinical_significance\n"
        "BRCA1\tP38398\t1JNX\tMDLSALRVEE...\tC61G\tPathogenic\n"
        "BRCA1\tP38398\t1JNX\tMDLSALRVEE...\tS1613G\tBenign\n"
    )
    print(f"[assets] wrote ClinVar schema template to {out}")
    print("[assets]   fill it in and save as missense_pairs.tsv to use data.source=clinvar")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--all", action="store_true", help="fetch models, PDB, corpus and ProteinGym")
    p.add_argument("--models", action="store_true", help="fetch ESM-2 checkpoints")
    p.add_argument("--model-ids", nargs="*", default=DEFAULT_MODELS)
    p.add_argument("--pdb", nargs="*", metavar="ID", help="PDB ids to download")
    p.add_argument("--corpus", action="store_true", help="fetch a UniProt FASTA for SAE training")
    p.add_argument("--corpus-size", type=int, default=10000)
    p.add_argument("--corpus-min-len", type=int, default=40)
    p.add_argument("--corpus-max-len", type=int, default=512)
    p.add_argument("--proteingym", action="store_true", help="fetch the ProteinGym benchmark")
    p.add_argument("--clinvar", action="store_true", help="write the ClinVar schema template")
    args = p.parse_args()

    if not any(
        [args.all, args.models, args.pdb, args.corpus, args.proteingym, args.clinvar]
    ):
        p.print_help()
        return 1

    ASSETS.mkdir(parents=True, exist_ok=True)

    if args.all or args.models:
        fetch_models(args.model_ids)
    if args.all or args.pdb is not None:
        fetch_pdb(args.pdb if args.pdb else DEFAULT_PDBS)
    if args.all or args.corpus:
        fetch_corpus(args.corpus_size, args.corpus_min_len, args.corpus_max_len)
    if args.all or args.proteingym:
        fetch_proteingym()
    if args.all or args.clinvar:
        write_clinvar_template()

    print("[assets] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
