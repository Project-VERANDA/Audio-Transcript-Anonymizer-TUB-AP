# ─── cas_to_report.py ──────────────────────────────────────────────────────
import json, itertools, pathlib

def cas_to_report(path: str) -> dict:
    """
    Convert a WebAnno/UIMA-CAS JSON 0.4.0 file into the `report`
    structure expected by dict_to_brat().
    """
    j = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))

    # 1) gather every feature structure (old + new layout)
    if "_referenced_fss" in j:
        fs_list = list(itertools.chain.from_iterable(j["_referenced_fss"].values()))
    else:                                   # your files
        fs_list = j["%FEATURE_STRUCTURES"]

    text   = next(fs for fs in fs_list if fs["%TYPE"] == "uima.cas.Sofa")["sofaString"]
    tokens = [fs for fs in fs_list if fs["%TYPE"].endswith(".Token")]
    spans  = [fs for fs in fs_list if fs["%TYPE"] == "custom.Span"]
    sents  = [fs for fs in fs_list if fs["%TYPE"].endswith(".Sentence")]

    span_map = {(s["begin"], s["end"]): s["label"] for s in spans}

    tokens.sort(key=lambda t: t["begin"])
    sents .sort(key=lambda s: s["begin"])

    report = {"filename": pathlib.Path(path).stem, "sentences": []}

    for sent in sents:
        in_sent = [tok for tok in tokens if sent["begin"] <= tok["begin"] < sent["end"]]
        sent_tokens, sent_labels, sent_offsets = [], [], []

        for tok in in_sent:
            word = text[tok["begin"]:tok["end"]]
            sent_tokens .append(word)
            sent_offsets.append([tok["begin"], tok["end"]])

            # default label
            lab = "O"
            for (s, e), phi in span_map.items():
                if s <= tok["begin"] < tok["end"] <= e:
                    lab = ("B-" if tok["begin"] == s else "I-") + phi
                    break
            sent_labels.append(lab)

        report["sentences"].append({
            "tokens":  sent_tokens,
            "labels":  sent_labels,
            "offset":  sent_offsets
        })

    return report
