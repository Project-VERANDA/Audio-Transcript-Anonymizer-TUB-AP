import os
import time
import ner_deidentify as RedactReplace
import spacy

# Define paths
base_path = "/Users/hamann/Documents/Uni/SoSe25/QU Project/group_github/anonymize-medical-data/anonymize-medical-data" # *** change main path to the location of this file (get current path by typing 'pwd' into your terminal)
annonym_folder = os.path.join(base_path, "annonym")
test_data_txt = os.path.join(base_path, "data/test_data")

# Create annonym folder if doesn't exist
if not os.path.exists(annonym_folder):
    os.makedirs(annonym_folder)

# start timer
tic = time.perf_counter()
nlp = spacy.load('output/model-last')

for file in os.listdir(test_data_txt):
    file_path = os.path.join(test_data_txt, file)
    if os.path.isfile(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

    doc = nlp(text)

    deidentified_doc = RedactReplace.deidentify_entities_in_doc(doc, redact_replace="REDACT")  # Use REDACT or REDACT_STAR as needed

    # Save the anonymized text to a new file
    anonymized_file_path = os.path.join(annonym_folder, os.path.splitext(file)[0] + "_anonymized.txt")
    with open(anonymized_file_path, 'w', encoding='utf-8') as f:
        f.write(str(deidentified_doc))

toc = time.perf_counter()  # end timer
print(f"De-identification and creation of anonymized files completed in {toc - tic:0.4f} seconds.")