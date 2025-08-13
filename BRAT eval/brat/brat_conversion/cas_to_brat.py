#!/usr/bin/env python3
"""
Convert one or many WebAnno/UIMA-CAS JSON files into Brat *.ann* standoff files.
This script was used to convert the WebAnno gold standard files into the Brat
format.(no predictions)
USAGE
-----
# single file
python cas_to_brat.py path/to/file.json --outdir data/brat/gold/

# every *.json under a folder (non-recursive)
python cas_to_brat.py path/to/folder --outdir data/brat/gold/

# recursive walk and keep directory structure under --outdir
python cas_to_brat.py path/to/folder --outdir data/brat/gold/ --recursive
"""
from __future__ import annotations
import argparse, pathlib, sys

# ---- your two helpers -------------------------------------------------------
from cas_to_report   import cas_to_report      # function we wrote last turn
from convert_to_brat import dict_to_brat       # provided by your supervisor
# -----------------------------------------------------------------------------


def iter_input_paths(path: pathlib.Path, recursive: bool):
    """Yield every *.json file to process."""
    if path.is_file():
        yield path
    elif path.is_dir():
        glob = "**/*.json" if recursive else "*.json"
        for p in path.glob(glob):
            if p.is_file():
                yield p
    else:
        sys.exit(f"[ERROR] {path} is neither file nor directory")

def fix_bio(tags):
    fixed, prev = [], 'O'
    for t in tags:
        if t.startswith('I-') and (prev == 'O' or prev[2:] != t[2:]):
            t = 'B-' + t[2:]
        fixed.append(t); prev = t
    return fixed


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("src",
                     help="Single CAS *.json* file OR a directory that "
                          "contains them")
    cli.add_argument("--outdir", "-o", default="data/brat/gold/",
                     help="Destination root for *.ann* files")
    cli.add_argument("--recursive", "-r", action="store_true",
                     help="Recurse into sub-directories when SRC is a folder")
    args = cli.parse_args()

    src_path  = pathlib.Path(args.src).expanduser().resolve()
    out_root  = pathlib.Path(args.outdir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    total, written = 0, 0
    for cas_file in iter_input_paths(src_path, args.recursive):
        total += 1
        # Preserve relative path under --outdir
        rel   = cas_file.relative_to(src_path) if src_path.is_dir() else cas_file.name
        brat_out = out_root / rel.with_suffix(".ann")
        brat_out.parent.mkdir(parents=True, exist_ok=True)

        report = cas_to_report(cas_file)
        for sent in report["sentences"]:
            sent["labels"] = fix_bio(sent["labels"])   # repair leading I- tags
        dict_to_brat(report, brat_dir=str(brat_out.parent))
        written += 1
        print(f"[OK]  {cas_file}  ➜  {brat_out}")

    print(f"Converted {written}/{total} file(s) into {out_root}")


if __name__ == "__main__":
    main()
