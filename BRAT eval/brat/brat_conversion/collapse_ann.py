#!/usr/bin/env python3
"""
Rewrite every entity type in one or many Brat *.ann files to a single tag.
This is useful to collapse all entity types into one, e.g. for evaluation

Example:
    python collapse_ann.py data/brat/gold LABEL
"""
import argparse, pathlib, re, sys

def process_file(path: pathlib.Path, new_tag: str):
    out_lines = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.startswith("T"):                 # entity line
                # T12  OLD_TAG  123 456  text
                parts = line.split("\t")
                span  = parts[1].split()
                span[0] = new_tag                    # replace type
                parts[1] = " ".join(span)
                line = "\t".join(parts)
            out_lines.append(line)
    path.write_text("".join(out_lines), encoding="utf-8")

def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("folder", help="directory that contains *.ann")
    cli.add_argument("tag",    help="new unified tag, e.g. LABEL")
    args = cli.parse_args()

    ann_dir = pathlib.Path(args.folder)
    for ann in ann_dir.glob("*.ann"):
        process_file(ann, args.tag)
    print(f"Collapsed all entity types in {ann_dir} to '{args.tag}'.")

if __name__ == "__main__":
    main()
