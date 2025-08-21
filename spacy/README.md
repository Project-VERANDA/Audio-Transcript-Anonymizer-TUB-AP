# Anonymize medical data pipeline

### Fine-tune the spacy model
- ``python SpacyNerTraining.py`` to create dev.spacy and train.spacy files
- Then train the model with ``python -m spacy train config.cfg --output ./output --paths.train ./train.spacy --paths.dev ./dev.spacy``

### Testing
``python main.py`` to anonymize the files in ``data/test_data/`` (in the real application with the whole whisperx pipeline this would be ``transcripts/``).

This will give you the annonymized files in the folder annonym. Also, you can find the entities in ``data/test_results/result_json``.
We can now compare the results with the entities in ``test_results/ground_truth_json`` that are taken from our annotations.

### Accuracy
The file ``accuracy.py`` compares the entities the spacy model found (see json files in ``data/test_results/result_json``) with our ground truth (see ``eval/test_results/evaluation/ground_truth_json``).
You can find the results of the evaluation in ``accuracy_logs_and_figures/``.

---
Make sure to change the paths in the following files:
- main - Line 15
- InputToTranscript - Line 4
- SpacyNerTest - Line 6
- SpacyNerTraining - Line 15

And change the use_auth_token in
- InputToTranscript - Line 71