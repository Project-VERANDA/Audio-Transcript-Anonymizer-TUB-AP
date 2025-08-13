#!/usr/bin/env python3
"""
Infer PHI spans for one or many WebAnno/UIMA-CAS JSON files and write them
as Brat *.ann* standoff files (via dict_to_brat).

USAGE
-----
# single file
python predict_to_brat.py best-model/ path/to/doc.json --outdir data/brat/preds/

# whole folder
python predict_to_brat.py best-model/ path/to/cas/dir --outdir data/brat/preds/ -r 
"""
from __future__ import annotations
import argparse, pathlib, itertools, torch
import sys
from time import perf_counter
_t0 = perf_counter()

# ── helpers you already have ────────────────────────────────────────────────
from cas_to_report   import cas_to_report   # builds report dict w/ tokens+offsets
from convert_to_brat import dict_to_brat    # writes *.ann*

def iter_input_paths(path: pathlib.Path, recursive: bool):
    """Yield every *.json file to process, preserving consistency with cas_to_brat."""
    if path.is_file():
        yield path
    elif path.is_dir():
        glob = "**/*.json" if recursive else "*.json"
        for p in path.glob(glob):
            if p.is_file():
                yield p
    else:
        sys.exit(f"[ERROR] {path} is neither file nor directory")

# ─────────────────────────────────────────────────────────────────────────────

def predict_sentence(tokens: list[str],
                     model,
                     tokenizer,
                     device: str = "cpu") -> list[str]:
    """
    Run the HF token-classifier on ONE sentence (list of words) and
    return word-level BIO tags aligned with the input tokens.
    """
    # ▸ 1) Encode **without** turning the BatchEncoding into a plain dict
    enc = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=False
    ).to(device)                    # BatchEncoding still has .word_ids()

    word_ids = enc.word_ids(batch_index=0)        # list[Optional[int]]
    with torch.no_grad():
        logits = model(**enc).logits              # (1, seq_len, num_labels)

    pred_ids = logits.argmax(-1).squeeze(0).tolist()   # sub-token ids
    id2label = model.config.id2label

    # ▸ 2) Collapse sub-tokens → one label per *word*
    word_labels, seen = [], set()
    for idx, wid in enumerate(word_ids):
        # skip special tokens and subsequent pieces of the same word
        if wid is None or wid in seen:
            continue
        seen.add(wid)
        word_labels.append(id2label[int(pred_ids[idx])])

    return word_labels        # len(word_labels) == len(tokens)

def fix_bio(tags):
    """
    Repair BIO sequence and collapse stray B-X inside an entity
    into I-X, so that dict_to_brat will emit a single span.
    """
    fixed, prev = [], 'O'
    for t in tags:
        # split into prefix + type
        if t == 'O':
            fixed.append(t); prev = t
            continue

        prefix, typ = t.split('-', 1)

        # 1) if I- appears without matching prev, make it B-
        if prefix == 'I' and (prev == 'O' or prev[2:] != typ):
            prefix = 'B'

        # 2) if B- appears but prev was same type, change to I-
        elif prefix == 'B' and (prev.startswith(('B-','I-')) and prev[2:] == typ):
            prefix = 'I'

        new_tag = f"{prefix}-{typ}"
        fixed.append(new_tag)
        prev = new_tag
    return fixed

def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("model", help="folder with fine-tuned checkpoint")
    cli.add_argument("cas",   help="CAS file OR directory")
    cli.add_argument("--outdir", "-o", default="data/brat/preds/")
    cli.add_argument("--device", default="cpu")
    cli.add_argument("--recursive", "-r", action="store_true")
    args = cli.parse_args()

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model     = AutoModelForTokenClassification.from_pretrained(args.model).to(device)
    model.eval()

    cas_root = pathlib.Path(args.cas).resolve()
    out_root = pathlib.Path(args.outdir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    cas_files = list(iter_input_paths(cas_root, args.recursive))

    for cas_file in cas_files:
        # preserve folder structure just like cas_to_brat
        rel      = cas_file.relative_to(cas_root) if cas_root.is_dir() else cas_file.name
        brat_dir = out_root / rel.parent
        brat_dir.mkdir(parents=True, exist_ok=True)

        report = cas_to_report(cas_file)      # use Path, not str, for consistency

        # ── run inference & repair tags ─────────────────────────
        for sent in report["sentences"]:
            pred_tags      = predict_sentence(sent["tokens"], model, tokenizer, device)
            sent["labels"] = fix_bio(pred_tags)
            try:
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()     # make sure any queued GPU kernels finish
            except ImportError:
                pass

            elapsed = perf_counter() - _t0
            print(f"[done] processed {__file__} in {elapsed:,.2f} s")

        # ── write Brat .ann just like cas_to_brat ────────────────
        dict_to_brat(report, brat_dir=str(brat_dir))
        print(f"[OK]  {cas_file}  ➜  {brat_dir / rel.with_suffix('.ann')}")

if __name__ == "__main__":
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    main()
