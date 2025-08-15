#!/usr/bin/env python3
"""
Run a spaCy NER model on raw transcript *.txt* files **plus** their matching
UIMA‑CAS *.json* files and write BRAT *.ann* standoff files whose character
offsets refer to the *sofaString* stored inside every CAS.

Why this script?
────────────────
Some transcripts contain a mixture of Windows (\r\n) and Unix (\n) line breaks.
When we feed the raw *.txt* into spaCy, the offsets it returns are relative to
that raw text – but the official gold offsets are defined on the CAS
*sofaString*.  We therefore re‑align every predicted span **exactly** the way
`essa.py` does, because that logic already proved to survive the tricky
\r/\n cases.

Usage (mirrors `predict_to_brat.py`)
────────────────────────────────────
python spacy_to_brat.py \
       /path/to/spacy/model  \
       /path/to/transcripts_dir  \
       --outdir data/brat/spacy/  [--recursive]

The script expects        myfile.txt
and the matching CAS at    myfile.json  (same basename, same folder).
It writes                    myfile.ann in --outdir, preserving any folder
structure when called with --recursive.
"""

from __future__ import annotations
import argparse, json, itertools, os, re, sys, typing
from pathlib import Path
import torch
import spacy
from time import perf_counter
_t0 = perf_counter()

# ───────────────────────── logging ───────────────────────────────────────────
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s ▶︎ %(message)s")
log = logging.getLogger(__name__)

# ───────────────────────── helper functions ─────────────────────────────────

def load_sofa_string(txt_path: Path) -> str:
    """Return the *sofaString* located in the sibling *.json* CAS file."""
    cas_path = txt_path.with_suffix(".json")
    if not cas_path.exists():
        raise FileNotFoundError(f"No matching CAS found for {txt_path}")

    j = json.loads(cas_path.read_text(encoding="utf-8"))

    # compatible with both the historical ("_referenced_fss") and current
    # ("%FEATURE_STRUCTURES") CAS layouts
    if "_referenced_fss" in j:                                # legacy layout
        fs_list = list(itertools.chain.from_iterable(j["_referenced_fss"].values()))
    elif "%FEATURE_STRUCTURES" in j:                          # modern layout
        fs_list = j["%FEATURE_STRUCTURES"]
    else:
        raise ValueError(f"Unsupported CAS layout: {cas_path}")

    sofa = next(fs for fs in fs_list if fs.get("%TYPE") == "uima.cas.Sofa")
    return sofa["sofaString"]


def compute_offsets(full_text: str, annotations: list[dict]) -> list[dict]:
    used = []
    for ann in annotations:
        span = ann['text']
        for m in re.finditer(re.escape(span), full_text):
            if not any(m.start() < u_end and m.end() > u_start for u_start, u_end in used):
                ann['start'], ann['end'] = m.start(), m.end()
                used.append((m.start(), m.end()))
                break
    return annotations


def iter_input_paths(path: Path, recursive: bool) -> typing.Iterable[Path]:
    """Yield every *.txt* (transcript) file under *path*."""
    if path.is_file() and path.suffix.lower() == ".txt":
        yield path
    elif path.is_dir():
        glob = "**/*.txt" if recursive else "*.txt"
        for p in path.glob(glob):
            if p.is_file():
                yield p
    else:
        sys.exit(f"[ERROR] {path} is neither a .txt file nor a directory")


# ───────────────────────── core pipeline ────────────────────────────────────

def process_file(txt_path: Path, nlp, out_root: Path, label_map: dict[str,str]):
    """Run spaCy on *txt_path* and write the corresponding *.ann* file."""
    raw_text  = txt_path.read_text(encoding="utf-8")
    sofa_text = load_sofa_string(txt_path)

    # 1) spaCy inference (could be a GPU model – let spaCy decide)
    doc = nlp(raw_text)
    
    annotations: list[dict] = []
    for ent in doc.ents:
        annotations.append({
            "label": label_map.get(ent.label_, ent.label_),
            "text":  ent.text,
            "start": ent.start_char,
            "end":   ent.end_char,
        })
        try:
                
            if torch.cuda.is_available():
                torch.cuda.synchronize()     # make sure any queued GPU kernels finish
        except ImportError:
            pass

        elapsed = perf_counter() - _t0
        print(f"[done] processed {__file__} in {elapsed:,.2f} s")
        #print(annotations[-1])  # Debug: print each annotation
        #print(sofa_text[ent.start_char : ent.end_char]) 
        '''for ent in doc.ents:
            surf = raw_text[ent.start_char : ent.end_char]
            print(f"{ent.label_:<8} {ent.start_char:>5}-{ent.end_char:<5} |",
                f"spaCy: {ent.text!r} | sofa slice: {surf!r}")'''

    # 2) Map every predicted span back to *sofa_text* offsets
    annotations = compute_offsets(sofa_text, annotations)
    annotations = [a for a in annotations if "start" in a and "end" in a]

    # 3) Brat writing – preserve folder structure under --outdir
    rel       = txt_path.relative_to(src_root) if src_root.is_dir() else txt_path.name
    ann_path  = (out_root / rel).with_suffix(".ann")
    ann_path.parent.mkdir(parents=True, exist_ok=True)

    with ann_path.open("w", encoding="utf-8") as out:
        for tid, ann in enumerate(sorted(annotations, key=lambda x: x["start"]), 1):
            clean = ann ["text"].replace("\n", " ").replace("\r", " ").replace("\t", " ")
            out.write(f"T{tid}\t{ann['label']} {ann['start']} {ann['end']}\t{clean}\n")

    log.info("✓ %s → %s (%d spans)", txt_path.name, ann_path, len(annotations))


# ───────────────────────── main ─────────────────────────────────────────────
if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("model",   help="spaCy model name or path (e.g. nlpcraft/de-id-ger)")
    cli.add_argument("src",     help="Transcript .txt file OR a directory that contains them")
    cli.add_argument("--outdir", "-o", default="data/brat/spacy/",
                     help="Destination root for *.ann* files")
    cli.add_argument("--recursive", "-r", action="store_true",
                     help="Recurse into sub-directories when SRC is a folder")
    cli.add_argument("--map", nargs="*", metavar="RULE", default=[],
                     help="Optional label remapping like PER=NAME_OTHER DATE=DATE")
    args = cli.parse_args()

    # label remapping dict (spaCy‑label → project‑label)
    label_map = {}
    for rule in args.map:
        try:
            src_lbl, tgt_lbl = rule.split("=", 1)
            label_map[src_lbl] = tgt_lbl
        except ValueError:
            sys.exit(f"[ERROR] --map rule '{rule}' is not of form SRC=TGT")

    log.info("Loading spaCy model … (%s)", args.model)
    nlp = spacy.load(args.model)

    src_root = Path(args.src).expanduser().resolve()
    out_root = Path(args.outdir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    files = list(iter_input_paths(src_root, args.recursive))
    if not files:
        sys.exit("[ERROR] No *.txt* files found – nothing to do")

    for fp in files:
        try:
            process_file(fp, nlp, out_root, label_map)
        except Exception as e:
            log.error("✗ %s – %s", fp.name, e)

    log.info("Finished – %d file(s) processed", len(files))
