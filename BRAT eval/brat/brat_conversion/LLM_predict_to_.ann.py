import json
import os
import re
import time
import requests
import logging
import argparse
import glob
import itertools
from pathlib import Path
from time import perf_counter
_t0 = perf_counter()

CHAT_AI_API_KEY = '6e3def45d61b0b20547a1fcbab6464d8'
CHAT_AI_ENDPOINT = 'https://chat-ai.academiccloud.de/v1'
DEFAULT_MODEL = os.getenv('CHAT_AI_MODEL', 'llama-3.1-8b-instruct')

AVAILABLE_MODELS = {
    'llama-3.1-sauerkrautlm-70b-instruct': 'Llama 3.1 SauerkrautLM 70B Instruct',
    'llama-3.3-70b-instruct':           'Meta Llama 3.3 70B Instruct',
    'qwen3-32b':                       'Qwen 3 32B',
    'qwen3-235b-a22b':                 'Qwen 3 235B A22B',
    'qwen2.5-coder-32b-instruct':      'Qwen 2.5 Coder 32B Instruct',
    'codestral-22b':                   'Mistral Codestral 22B',
    'qwq-32b':                         'Qwen QwQ 32B',
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_sofa_string(txt_path: str) -> str:
    cas_path = Path(txt_path).with_suffix('.json')
    j = json.loads(cas_path.read_text(encoding='utf-8'))

    # replicate cas_to_report.py logic for old vs. new CAS layout
    if '_referenced_fss' in j:
        fs_list = list(itertools.chain.from_iterable(j['_referenced_fss'].values()))
    elif '%FEATURE_STRUCTURES' in j:
        fs_list = j['%FEATURE_STRUCTURES']
    else:
        raise ValueError(f"Unsupported CAS layout in file: {cas_path}")

    sofa_obj = next(fs for fs in fs_list if fs.get('%TYPE') == 'uima.cas.Sofa')
    return sofa_obj['sofaString']


def compute_offsets(full_text: str, annotations: list[dict]) -> list[dict]:
    used = []
    for ann in annotations:
        span = ann['text']
        for m in re.finditer(re.escape(span), full_text):
            if not any(m.start() < u_end and m.end() > u_start for u_start, u_end in used):
                ann['start'], ann['end'] = m.start(), m.end()
                used.append((m.start(), m.end()))
                break
    return annotations


def anonymize_text(file_path: str, selected_model: str):
    try:
        # 1) Load the sofaString (so offsets match gold)
        text = load_sofa_string(file_path)

        # 2) Resolve model alias
        name_to_key = {v: k for k, v in AVAILABLE_MODELS.items()}
        if selected_model in name_to_key:
            selected_model = name_to_key[selected_model]
        if selected_model not in AVAILABLE_MODELS:
            return {'error': f'Invalid model selected: {selected_model}'}, 400
        model_name = AVAILABLE_MODELS[selected_model]

        # 3) Prepare prompt + headers
        prompt =  """ You are a medical de-identification assistant.  When I give you a German transcript, you must return exactly one JSON object, nothing else, matching this schema:

            {
            "name": "annotate_deid",
            "arguments": {
                "annotations": [
                {
                    "label": "<ENTITY_TYPE>",
                    "text": "<exact span>"
                },
                …
                ]
            }
            }

            Use these rules to anonymize (replace) only the indicated PHI types—leave everything else (including SPEAKER_00, SPEAKER_01, punctuation, line breaks) verbatim:

            - Patient names → [NAME_PATIENT]  
            - Doctor names including medical titles → [NAME_DOCTOR]  
            - Other names → [NAME_OTHER]  
            - Usernames → [NAME_USERNAME]
            - Professions → [PROFESSION]  
            - Phone numbers → [CONTACT_PHONE]  
            - Email addresses → [CONTACT_EMAIL] 
            - Fax numbers → [CONTACT_FAX]
            - URLs → [CONTACT_URL] 
            - Street addresses → [LOCATION_STREET] 
            - Cities → [LOCATION_CITY] 
            - Countries → [LOCATION_COUNTRY] 
            - ZIP codes → [LOCATION_ZIP]  
            - Hospitals → [LOCATION_HOSPITAL]  
            - Organizations → [LOCATION_ORGANISATION]  
            - Specific dates like 12.03 or 14.10.2005 → [DATE]  
            - Ages → [AGE]  
            - ID numbers → [ID]  

            ⚠️ *CRITICAL*:  
            • Do NOT modify or remove speaker tags like SPEAKER_00, SPEAKER_01, etc.  
            • Keep all text in German—do not translate.  
            • Output must be strictly valid JSON with no extra commentary or markdown.

            Here is the transcript:
                    """  
        headers = {
            'Authorization': f'Bearer {CHAT_AI_API_KEY}',
            'Content-Type': 'application/json'
        }
        payload = {
            "model": selected_model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user",   "content": f"Text to anonymize:\n\n{text}"}
            ],
            "max_tokens": 8000,
            "temperature": 0.1
        }

        # 4) Call API with retries
        max_retries, timeout = 3, 360
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{CHAT_AI_ENDPOINT}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=timeout
                )
                break
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    logger.warning(f"Timeout, retry {attempt+1}...")
                    time.sleep(2)
                    continue
                return {'error': 'Anonymization service slow.'}, 504
            except requests.exceptions.RequestException as e:
                return {'error': f'Connection error: {e}'}, 503

        if response.status_code != 200:
            return {'error': f'API error: {response.status_code}'}, 500

        resp = response.json()
        content = resp['choices'][0]['message']['content'].strip()
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        try:
            anonymized_json = json.loads(content)
        except json.JSONDecodeError:
            return {'error': 'Invalid JSON from model', 'details': content}, 500

        # 5) Compute offsets ONCE, using sofaString
        annots = anonymized_json['arguments']['annotations']
        annots = compute_offsets(text, annots)
        anonymized_json['arguments']['annotations'] = annots

        return {'success': True, 'anonymized_text': anonymized_json, 'model_used': model_name}, 200

    except Exception as e:
        logger.error(f"Error anonymizing text: {e}")
        return {'error': str(e)}, 500


def json_to_brat_ann(json_data: dict, txt_path: str, ann_path: str):
    """
    Write ONLY T-lines, properly tab-separated for Spradie’s eval.
    Format per line:
      T{id}  {label}  {start} {end}  {text}
    """
    sofa = load_sofa_string(txt_path)
    # compute_offsets once on the sofaString for perfect alignment
    annots = [
        a for a in compute_offsets(sofa, json_data['anonymized_text']['arguments']['annotations'])
        if 'start' in a and 'end' in a
    ]

    os.makedirs(os.path.dirname(ann_path), exist_ok=True)
    with open(ann_path, 'w', encoding='utf-8') as out:
        # sort by start offset so T-IDs are in document order
        for tid, ann in enumerate(sorted(annots, key=lambda x: x['start']), start=1):
            out.write(
                f"T{tid}\t{ann['label']} {ann['start']} {ann['end']}\t{ann['text']}\n"
            )


def process_folder(input_dir: str, output_dir: str, model: str):
    os.makedirs(output_dir, exist_ok=True)
    for txt_path in glob.glob(os.path.join(input_dir, "*.txt")):
        base     = os.path.splitext(os.path.basename(txt_path))[0]
        ann_path = os.path.join(output_dir, f"{base}.ann")

        result, status = anonymize_text(txt_path, model)
        if status != 200:
            logger.error("Failed to anonymize %s: %s", txt_path, result)
            continue

        json_to_brat_ann(result, txt_path, ann_path)
        print(f"→ Written BRAT .ann: {ann_path}")

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()     # make sure any queued GPU kernels finish
        except ImportError:
            pass

        elapsed = perf_counter() - _t0
        print(f"[done] processed {__file__} in {elapsed:,.2f} s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir",  help="folder with .txt transcripts")
    parser.add_argument("output_dir", help="where to write the .ann files")
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL,
        choices=list(AVAILABLE_MODELS.keys())
    )
    args = parser.parse_args()
    process_folder(args.input_dir, args.output_dir, args.model)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()     # make sure any queued GPU kernels finish
    except ImportError:
            pass

    elapsed = perf_counter() - _t0
    print(f"[done] processed {__file__} in {elapsed:,.2f} s")