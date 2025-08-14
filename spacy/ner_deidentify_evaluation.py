import spacy
import json
import os
import re

# define paths
result_path = "eval/test_results/evaluation/result_json/"

nlp = spacy.load('output/model-last')

##############################################################
# REDACT MENU
##############################################################
global known_names
known_names = []
def redact_entity(entity_type, token):
    if entity_type == "NAME":
        check = hash(str(token))
        if check not in known_names:
            known_names.append(check)
            redacted_text = "[REDACTED_" + entity_type + "0" + str(len(known_names)) + "]"
        else:
            known_number = known_names.index(check)
            redacted_text = "[REDACTED_" + entity_type + "0" + str(known_number) + "]"
    else:
        redacted_text = "[REDACTED_" + entity_type + "]"

    return redacted_text

def redact_entities_with_star(token):
    return "*" * len(token.text)


##############################################################
# FIND ELEMENTS THAT ARE SENSITIVE AND REDACT THEM
##############################################################
# https://realpython.com/natural-language-processing-spacy-python/
def deidentify_entities_in_doc(file_name, text, redact_replace):
    known_names.clear()

    nlp_doc = nlp(text)
    with nlp_doc.retokenize() as retokenizer:
        for ent in nlp_doc.ents:
            retokenizer.merge(ent)
    tokens = []

    ### for stats ###
    dictionary = {
        "begin": 0,
        "end": 0,
        "label": "",
    }
    result_file = result_path + file_name.split(".")[0] + ".json"
    #################

    ### for stats ###
    with open(result_file, "w") as outfile:
        outfile.write("[\n")
        #################
        for token in nlp_doc:
            # AGE PATTERN MATCHING
            if token.ent_type_ != "AGE":
                if token.i < len(nlp_doc) - 1:
                    next_token = nlp_doc[token.i + 1]
                    if re.fullmatch(r"Jahre(n)?", next_token.text) and re.fullmatch(r"\d{2,3}", token.text) and token.text[0] != "0":
                        token.ent_type_ = "AGE"
                        # print(f"Found scheme: {token.text} with next token {next_token.text} at position {token.idx} in {result_file}")

            # LOCATION_ZIP PATTERN MATCHING
            if token.ent_type_ != "LOCATION_ZIP":
                if re.fullmatch(r"\d{5}.*", token.text) and token.text[0] != "0":
                    token.ent_type_ = "LOCATION_ZIP"
                    # print(f"Found scheme: {token.text} at position {token.idx} in {result_file}")

            # CONTACT PHONE PATTERN MATCHING
            if token.ent_type_ != "CONTACT_PHONE":
                if re.fullmatch(r"\d{6,20}.*", token.text):
                    token.ent_type_ = "CONTACT_PHONE"
                    # print(f"Found scheme: {token.text} at position {token.idx} in {result_file}")

            # NER WITH SPACY
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
                
                ### for stats ###
                dictionary["begin"] = token.idx
                dictionary["end"] = token.idx + len(token.text)
                dictionary["label"] = token.ent_type_
                json_object = json.dumps(dictionary, indent=4)
                outfile.write(json_object)
                outfile.write(",")
                #################
            else:
                tokens.append(token.text) #.text_with_ws)

        ### for stats ###
        outfile.seek(outfile.tell() - 1, os.SEEK_SET)
        outfile.write("\n]")
        #################

    return nlp(" ".join(tokens))