import os
import string

_WS = set(string.whitespace)          # {' ', '\n', '\t', …}
MERGE_LABELS = {}               # extend if you need more

def _merge_adjacent_whitespace(entities, full_text):
    merged = []
    for ent in entities:
        if merged and ent[1] == merged[-1][1] and ent[1] in MERGE_LABELS:
            prev = merged[-1]
            gap  = full_text[prev[3]:ent[2]]       # chars between spans
            if gap and all(c in _WS for c in gap): # ← ONLY whitespace
                merged.pop()
                new_ent = (
                    prev[0], prev[1], prev[2], ent[3],
                    full_text[prev[2]:ent[3]]       # exact joined text
                )
                merged.append(new_ent)
                continue
        merged.append(ent)
    return merged

def dict_to_brat(
    report: dict,
    label_column_name: str = "labels",
    brat_dir: str = '.'
):
    """
    Convert a single report (with tokens, labels, offsets) into a BRAT .ann file.

    report: dict containing "filename" and "sentences" (each with tokens, labels, offset)
    label_column_name: column for NER labels (e.g. "labels" or "preds")
    brat_dir: directory to store .ann files
    """
    filename   = report["filename"]
    sentences  = report["sentences"]

    # Reconstruct full text from offsets
    full_text_chars = []
    last_char_pos   = 0
    for sent in sentences:
        for tok, off in zip(sent.get("tokens", []), sent.get("offset", [])):
            b, e = map(int, off)
            full_text_chars.append(" " * (b - last_char_pos))
            full_text_chars.append(tok)
            last_char_pos = e
    full_text = "".join(full_text_chars)

    # Prepare to collect sentences and entities
    all_entities      = []
    sentence_texts    = []
    sentence_ranges   = []  # (start_char, end_char) per sentence
    entity_count      = 0

    for sent in sentences:
        # determine char span for this sentence
        offsets = sent.get("offset", [])
        if offsets:
            sent_start = int(offsets[0][0])
            sent_end   = int(offsets[-1][1])
        else:
            sent_start = sent_end = 0
        sentence_ranges.append((sent_start, sent_end))

        # build sentence text and collect entities
        sentence_text = ""
        entities      = []
        entity_text   = ""
        entity_type   = None
        entity_start  = None
        entity_end    = None
        last_offset   = None

        for token, label, offset in zip(
            sent.get("tokens", []),
            sent.get(label_column_name, []),
            sent.get("offset", [])
        ):
            b, e = map(int, offset)
            # build sentence text
            if last_offset is None:
                sentence_text += token
            else:
                gap = b - last_offset
                sentence_text += " " * gap + token
            last_offset = e

            # entity logic
            if label.startswith("B-"):
                if entity_text:
                    entities.append((entity_count, entity_type, entity_start, entity_end, entity_text))
                entity_count += 1
                entity_type  = label.split("-", 1)[1]
                entity_start = b
                entity_end   = e
                entity_text  = token
            elif label.startswith("I-") and entity_text:
                if b > entity_end:
                    entity_text += full_text[entity_end:b]
                entity_end  = e
                entity_text += token
            else:
                if entity_text:
                    entities.append((entity_count, entity_type, entity_start, entity_end, entity_text))
                    entity_text  = ""
                    entity_type  = None
                    entity_start = None
                    entity_end   = None
        # flush last entity in this sentence
        if entity_text:
            entities.append((entity_count, entity_type, entity_start, entity_end, entity_text))

        sentence_texts.append(sentence_text)
        all_entities.extend(entities)

    # Merge adjacent whitespace globally
    merged = _merge_adjacent_whitespace(sorted(all_entities, key=lambda x: x[2]), full_text)

    # Assign each merged entity to the sentence containing its start offset
    sent_ent_map = {i: [] for i in range(len(sentence_ranges))}
    for ent in merged:
        ent_id, ent_type, s, e, text = ent
        assigned = False
        for idx, (start_char, end_char) in enumerate(sentence_ranges):
            if s >= start_char and s < end_char:
                sent_ent_map[idx].append(ent)
                assigned = True
                break
        if not assigned:
            # fallback: put at first sentence
            sent_ent_map[0].append(ent)

    # Write BRAT file with inline entity annotations per sentence
    os.makedirs(brat_dir, exist_ok=True)
    brat_path = os.path.join(brat_dir, f"{os.path.splitext(filename)[0]}.ann")
    with open(brat_path, "w", encoding="utf-8") as f:
        # optional header with full text
        f.write(f"#0\t{filename} {full_text}\n")
        # write each sentence and its assigned entities
        for idx, st in enumerate(sentence_texts):
            f.write(f"#{idx+1}\t{st}\n")
            for ent_id, ent_type, s, e, text in sent_ent_map[idx]:
                f.write(f"T{ent_id}\t{ent_type} {s} {e}\t{text}\n")
