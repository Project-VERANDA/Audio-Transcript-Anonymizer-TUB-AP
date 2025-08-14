import json
import os
import matplotlib.pyplot as plt
from collections import Counter

DATA_CHOICE = "evaluation"  # Change to "grascco" if using Grasco data or "interviews" for interview data
EVALUATION_LOG = "evaluation_log_250720.txt"    # Change to your desired log file name

# Define paths
base_path = "/Users/hamann/Documents/Uni/SoSe25/QU Project/group_github/anonymize-medical-data/anonymize-medical-data" # *** change main path to the location of this file (get current path by typing 'pwd' into your terminal)
ground_truth = os.path.join(base_path, "eval/test_results/" + DATA_CHOICE + "/ground-truth-json")
test_data_results = os.path.join(base_path, "eval/test_results/" + DATA_CHOICE + "/result_json")
accuracy_log = os.path.join(base_path, "eval/accuracy_logs_and_figures/" + EVALUATION_LOG)


tp_entities = 0
fp_entities = 0
fn_entities = 0

file_names = []
tp_list = []
fp_list = []
fn_list = []

types_of_wrong_results_fp = Counter()
types_of_wrong_results_fn = Counter()


with open(accuracy_log, "w") as log_file:
    log_file.write("Accuracy Log\n")
    log_file.write("========================================================================\n\n")

    for file in os.listdir(test_data_results):

        log_file.write(f"----------------- File: {file} --------------------------------------------------------------\n")

        tp_entities_per_file = 0
        fp_entities_per_file = 0
        fn_entities_per_file = 0

        if file.endswith(".json"):
            results_path = os.path.join(test_data_results, file)
            if DATA_CHOICE == "grascco":
                filename = file.split(".")[0] + ".txt_phi.json"
            else:
                filename = file
            ground_truth_path = os.path.join(ground_truth, filename)

            with open(results_path, 'r') as results_file:
                results_data = json.load(results_file)

            if os.path.isfile(ground_truth_path):
                with open(ground_truth_path, 'r') as guide_file:
                    ground_truth_data = json.load(guide_file)
            else:
                print(f"Warning: Guide file {ground_truth_path} does not exist. Skipping...")
                continue

            # get the results data with begin, end and label
            result_entities = []
            for entity in results_data:
                if "_" in entity["label"] and entity["label"] != "NAME_TITLE":
                    entity_label = entity["label"].split("_")[0]
                    if entity_label == "LOC":
                        entity_label = "LOCATION"
                else:
                    entity_label = entity["label"]
                dictionary = {
                    "begin": entity["begin"],
                    "end": entity["end"],
                    "label": entity_label
                }
                result_entities.append(dictionary)

            # get the ground truth data with begin, end and label
            ground_truth_entities = []
            for ground_entity in ground_truth_data["%FEATURE_STRUCTURES"]:
                if ground_entity["%TYPE"] == "custom.Span" or ground_entity["%TYPE"] == "webanno.custom.PHI":
                    if "label" in ground_entity:
                        if "_" in ground_entity["label"] and ground_entity["label"] != "NAME_TITLE":
                            ground_entity_label = ground_entity["label"].split("_")[0]
                            if ground_entity_label == "LOC":
                                ground_entity_label = "LOCATION"
                        else:
                            ground_entity_label = ground_entity["label"]
                    elif "kind" in ground_entity:
                        if "_" in ground_entity["kind"] and ground_entity["kind"] != "NAME_TITLE":
                            ground_entity_label = ground_entity["kind"].split("_")[0]
                            if ground_entity_label == "LOC":
                                ground_entity_label = "LOCATION"
                        else:
                            ground_entity_label = ground_entity["kind"]
                
                    dictionary = {
                        "begin": ground_entity["begin"],
                        "end": ground_entity["end"],
                        "label": ground_entity_label
                    }
                    ground_truth_entities.append(dictionary)

            # compare results with ground truth to get true positives, false positives and false negatives
            for result in result_entities:
                tp = next(
                    (d for d in ground_truth_entities
                    if abs(result["begin"] - d["begin"]) <= 5
                    and abs(result["end"] - d["end"]) <= 5
                    and result["label"] == d["label"]),
                    None
                )
                if tp:
                    tp_entities_per_file += 1
                    tp_entities += 1
                    log_file.write(f"TP entity: {result} with ground truth: {tp} in file {file}\n")
                else:
                    fp_entities_per_file += 1
                    fp_entities += 1
                    log_file.write(f"FP entity: {result} with ground truth: {tp} in file {file}\n")

                    types_of_wrong_results_fp[result["label"]] += 1

            for gt in ground_truth_entities:
                fn = next(
                    (d for d in result_entities
                    if abs(gt["begin"] - d["begin"]) <= 5
                    and abs(gt["end"] - d["end"]) <= 5
                    and gt["label"] == d["label"]),
                    None
                )
                if not fn:
                    fn_entities_per_file += 1
                    fn_entities += 1
                    log_file.write(f"FN entity: {gt} in file {file}\n")

                    types_of_wrong_results_fn[gt["label"]] += 1
                        
            f_score_per_file = ((2 * tp_entities_per_file) / ((2 * tp_entities_per_file) + fp_entities_per_file + fn_entities_per_file)) * 100

            log_file.write("\n")
            log_file.write(f"Correct entities: {tp_entities_per_file} in file {file}\n")
            log_file.write(f"False positive entities: {fp_entities_per_file} in file {file}\n")
            log_file.write(f"Missed entities: {fn_entities_per_file} in file {file}\n")
            log_file.write(f"-> F-Score in {file}: {f_score_per_file}\n")
            log_file.write("\n")

            file_names.append(file)
            tp_list.append(tp_entities_per_file)
            fp_list.append(fp_entities_per_file)
            fn_list.append(fn_entities_per_file)

    # F-Score total
    f_score = ((2 * tp_entities) / ((2 * tp_entities) + fp_entities + fn_entities)) * 100
    log_file.write("\n\n")
    log_file.write("Final Results\n")
    log_file.write("========================================================================\n")
    log_file.write(f"F-Score total: {f_score}%")

    # make matplotlib chart with all the results
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.invert_yaxis()
    bar1 = ax.barh(file_names, tp_list, label='TP', color='green')
    bar2 = ax.barh(file_names, fp_list, left=tp_list, label='FP', color='pink')
    bar3 = ax.barh(file_names, fn_list, left=[i+j for i, j in zip(tp_list, fp_list)], label='FN', color='red')

    ax.set_xlabel('Amount of Entities')
    ax.set_title('TP/FP/FN Distribution per File')
    ax.legend()
    plt.tight_layout()
    plt.show()

    # further analysis of false positives and false negatives
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(types_of_wrong_results_fp.keys(), types_of_wrong_results_fp.values(), color='pink')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height / 2, str(height), ha='center', va='bottom')
    ax.set_xlabel('Entity Type')
    ax.set_ylabel('Count')
    ax.set_title('False Positive Entities by Type - Entities that were not in the ground truth but predicted by the model')
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(types_of_wrong_results_fn.keys(), types_of_wrong_results_fn.values(), color='red')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height / 2, str(height), ha='center', va='bottom')
    ax.set_xlabel('Entity Type')
    ax.set_ylabel('Count')
    ax.set_title('False Negative Entities by Type - Entities that were missed by the model')
    plt.tight_layout()
    plt.show()
