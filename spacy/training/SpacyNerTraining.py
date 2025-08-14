import json
import os
import re
import spacy
from spacy.tokens import DocBin
from tqdm import tqdm
from spacy.util import filter_spans
from collections import Counter
import time
import random

##################################################
# spacy model: de_core_news_lg
# https://spacy.io/models/de#de_core_news_lg
##################################################

# Define paths
base_path = "/Users/hamann/Documents/Uni/SoSe25/QU Project/group_github/anonymize-medical-data/anonymize-medical-data" # *** change main path to the location of this file (get current path by typing 'pwd' into your terminal)
training_txt_folder = os.path.join(base_path, "data/training_data/txt_training")
training_json_folder = os.path.join(base_path, "data/training_data/inception")

# TIMER
tic = time.perf_counter()

# --------------------------------------------------------------------------------------------------
#       get the training data from the json files and txt files
# --------------------------------------------------------------------------------------------------
training_data = []

for file in os.listdir(training_json_folder):
    temp_dict = {}
    temp_dict["entities"] = []

    # get the json files for the annotations
    with open(training_json_folder + "/" + file, "r", encoding='utf-8') as f:
        data = json.load(f)

    # get the text files
    currFile = file.split(".")[0] + ".txt"
    if not os.path.exists(training_txt_folder + "/" + currFile):
        print(f"Attention: Text file {currFile} does not exist in {training_txt_folder}. Skipping...")
        continue
    with open(training_txt_folder + "/" + currFile, "r", encoding='utf-8') as f:
        document = f.read()

    # create a dictionary with the text of the document and its entities
    temp_dict["text"] = document
    for structure in data["%FEATURE_STRUCTURES"]:
        if structure["%TYPE"] == "custom.Span" or structure["%TYPE"] == "webanno.custom.PHI":
            start = structure["begin"]
            end = structure["end"]
            text = document[start:end]
            if "label" in structure:
                label = structure["label"]
            elif "kind" in structure:
                label = structure["kind"]
            else:
                print("Attention: Label not specified")
                continue

            temp_dict["entities"].append((start, end, label, text))

    # create training data
    training_data.append(temp_dict)

# --------------------------------------------------------------------------------------------------
#       stats
# --------------------------------------------------------------------------------------------------  
print(f"Total number of files processed: {len(training_data)}")
print(f"Potential number of annotation files: {len(os.listdir(training_json_folder))}")
print(f"Potential number of txt files: {len(os.listdir(training_txt_folder))}")
print(f"Number of labels for each file: {[len(training_data[i]['entities']) for i in range(len(training_data))]}")
print(f"Total number of labels: {sum([len(training_data[i]['entities']) for i in range(len(training_data))])}")

# --------------------------------------------------------------------------------------------------
#       create validation data by randomly selecting 20% of the training data
# --------------------------------------------------------------------------------------------------
random.seed(42)
num_validation = len(training_data) // 5
validation_data = random.sample(training_data, num_validation)
training_data = [ex for ex in training_data if ex not in validation_data]

print("Train entities:", Counter([e[2] for ex in training_data for e in ex["entities"]]))
print("Val entities:", Counter([e[2] for ex in validation_data for e in ex["entities"]]))

# --------------------------------------------------------------------------------------------------
#       clean the entities
# --------------------------------------------------------------------------------------------------
def clean_entity_spans(text, entities):
    cleaned = []
    text_len = len(text)
    for start, end, label in entities:
        if start < 0 or end > text_len or start >= end or not label:
            continue
        while start < end and start < text_len and text[start].isspace():
            start += 1
        while end > start and end <= text_len and text[end - 1].isspace():
            end -= 1
        if start < end and text[start:end].strip():
            cleaned.append((start, end, label))
    return cleaned

# --------------------------------------------------------------------------------------------------
#   convert the data to the format required by spacy
#   https://spacy.io/usage/training#training-data
# --------------------------------------------------------------------------------------------------
def create_spacy_data(data, path):
    nlp = spacy.blank("de")
    doc_bin = DocBin()

    missed_spans = 0

    for example in tqdm(data):
        text = example["text"]

        labels = clean_entity_spans(text, example["entities"])
        doc = nlp.make_doc(text)

        ents = []
        for start, end, label in labels:
            if (start < end) and label:
                span = doc.char_span(start, end, label=label, alignment_mode="expand")
                if span is None:
                    missed_spans += 1
                else:
                    ents.append(span)
            else:
                print(f"Skipping entity with invalid start or end position or label: {start}, {end}, {label}")

        filtered_ents = filter_spans(ents)
        doc.ents = filtered_ents
        doc_bin.add(doc)

    doc_bin.to_disk(path) 

create_spacy_data(training_data, os.path.join(base_path, "train.spacy"))
create_spacy_data(validation_data, os.path.join(base_path, "dev.spacy"))

# TIMER
toc = time.perf_counter()  # end timer
print(f"Time taken to process the data and create train.spacy + dev.spacy: {toc - tic:0.4f} seconds")

##############################################################
# run this script to create train.spacy and dev.spacy files
# then train the model with the following command:
##############################################################
# > python SpacyNerTraining.py
# > python -m spacy train config.cfg --output ./output --paths.train ./train.spacy --paths.dev ./dev.spacy