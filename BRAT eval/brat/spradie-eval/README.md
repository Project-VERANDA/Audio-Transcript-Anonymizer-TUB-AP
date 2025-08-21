# SpRadIE: Information Extraction from Spanish radiology reports

Evaluation code for the SpRadIE challenge.

* [Home page](https://sites.google.com/view/spradie-2020)
* [Evaluation description](https://sites.google.com/view/spradie-2020/evaluation)


## Usage

To evaluate on a single file use:

```bash
python eval.py <source.txt> <reference.ann> <predicted.ann> [<entity_type>]
```

The entity type is optional. If not given, results are printed for all entity types.

To evaluate on a complete dataset use:

```bash
python eval_dir.py <ref_dir> <pred_dir> [<entity_type>]
```
