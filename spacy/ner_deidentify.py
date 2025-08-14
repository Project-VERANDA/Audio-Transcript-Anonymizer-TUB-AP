import spacy
import json
import os
import re

# define paths
base_path = "/Users/hamann/Documents/Uni/SoSe25/QU Project/group_github/anonymize-medical-data/anonymize-medical-data"  # *** change main path to the location of this file (get current path by typing 'pwd' into your terminal)
result_path = "test_results/"

nlp = spacy.load('output/model-last')

# --------------------------------------------------------------------------------------------------
# REDACT MENU
# --------------------------------------------------------------------------------------------------
global known_names
known_names = []

# redact with [REDACTED_LABEL]
def redact_entity(entity_type, token):
    if entity_type == "NAME":
        check = hash(str(token))
        if check not in known_names:
            known_names.append(check)
            redacted_text = "[" + entity_type + "0" + str(len(known_names)) + "]"
        else:
            known_number = known_names.index(check)
            redacted_text = "[" + entity_type + "0" + str(known_number) + "]"
    else:
        redacted_text = "[" + entity_type + "]"

    return redacted_text

# redact with *****
def redact_entities_with_star(token):
    return "*" * len(token.text)

# --------------------------------------------------------------------------------------------------
# FIND ELEMENTS THAT ARE SENSITIVE AND REDACT THEM
# https://realpython.com/natural-language-processing-spacy-python/
# --------------------------------------------------------------------------------------------------

def deidentify_entities_in_doc(file_name, nlp_doc, redact_replace):
    known_names.clear()

    with nlp_doc.retokenize() as retokenizer:
        for ent in nlp_doc.ents:
            retokenizer.merge(ent)
    tokens = []

    for token in nlp_doc:
        if token.ent_type_:
            # handle labels with underscores
            label_type = token.ent_type_
            if "_" in label_type and label_type != "NAME_TITLE":
                label_type = label_type.split("_")[0]
                if label_type == "LOC":
                    label_type = "LOCATION"

            # redact with [REDACTED_...] or with stars
            if redact_replace == "REDACT":
                tokens.append(redact_entity(label_type, token))
            elif redact_replace == "REDACT_STAR":
                tokens.append(redact_entities_with_star(token))
            else:
                raise ValueError("Invalid redact_replace option. Use 'REDACT' or 'REDACT_STAR'.")
        else:
            tokens.append(token.text)

    return nlp(" ".join(tokens))