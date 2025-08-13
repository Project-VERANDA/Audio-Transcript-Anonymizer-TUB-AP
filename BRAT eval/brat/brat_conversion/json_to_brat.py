#!/usr/bin/env python3
"""
Convert a folder of SpaCy-style plain JSON span files into Brat .ann files
(with only label and character offsets, no span text).

USAGE
-----
# Non-recursive (just one directory):
python spacy_json_folder_to_brat.py \
    --input_dir path/to/json_folder \
    --out_dir path/to/brat_folder

# Recursive (subdirectories too):
python spacy_json_folder_to_brat.py \
    --input_dir path/to/json_root \
    --out_dir path/to/brat_root \
    --recursive
"""
import json
import os
import argparse
from pathlib import Path

def convert_json_to_ann(json_path: Path, out_dir: Path):
    """
    Read a SpaCy JSON of spans [{"begin":…, "end":…, "label":…}, …]
    and write a Brat .ann file with lines:
      T1<TAB>LABEL begin end
      T2<TAB>LABEL begin end
    """
    spans = json.loads(json_path.read_text(encoding="utf-8"))
    # ensure output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)
    ann_path = out_dir / (json_path.stem + ".ann")

    with ann_path.open("w", encoding="utf-8") as out:
        for i, ent in enumerate(sorted(spans, key=lambda e: e["begin"]), start=1):
            b, e, lbl = ent["begin"], ent["end"], ent["label"]
            out.write(f"T{i}\t{lbl} {b} {e}\n")

    print(f"Wrote {ann_path} ({len(spans)} entities)")

def main():
    p = argparse.ArgumentParser(
        description="Convert a directory of SpaCy JSON spans → Brat .ann files"
    )
    p.add_argument(
        "--input_dir", "-i", required=True,
        help="Directory containing SpaCy JSON files"
    )
    p.add_argument(
        "--out_dir", "-o", required=True,
        help="Directory to write .ann files into"
    )
    p.add_argument(
        "--recursive", "-r", action="store_true",
        help="Recurse into subdirectories of input_dir"
    )
    args = p.parse_args()

    input_path = Path(args.input_dir)
    out_root    = Path(args.out_dir)

    if not input_path.exists():
        p.error(f"input_dir {input_path!r} does not exist")

    # gather all .json files
    if input_path.is_file():
        json_files = [input_path]
    else:
        pattern = "**/*.json" if args.recursive else "*.json"
        json_files = list(input_path.glob(pattern))

    if not json_files:
        print("No JSON files found. Exiting.")
        return

    for json_path in json_files:
        # preserve subdirectory structure relative to input_dir
        if input_path.is_dir():
            rel = json_path.relative_to(input_path).parent
            target_dir = out_root / rel
        else:
            target_dir = out_root

        convert_json_to_ann(json_path, target_dir)

if __name__ == "__main__":
    main()
