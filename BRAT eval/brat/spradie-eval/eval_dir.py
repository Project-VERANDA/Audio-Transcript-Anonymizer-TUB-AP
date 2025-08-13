import sys
import os
import logging
from collections import defaultdict

from ann import read_ann
from eval import eval, prec_rec_f1


if __name__ == '__main__':
    # uncomment for debugging output to stderr:
    # logging.basicConfig(level=logging.DEBUG)

    argv = sys.argv
    if len(argv) < 3:
        print('Usage: python eval_dir.py <ref_dir> <pred_dir> [<entity_type>]')
    else:
        if len(argv) == 4:
            etypes = [argv[3]]
        else:
            etypes = [
                'AGE',
                'CONTACT_EMAIL',
                'CONTACT_FAX',
                'CONTACT_PHONE',
                'CONTACT_URL',
                'DATE',
                'ID',
                'LOCATION_CITY',
                'LOCATION_HOSPITAL',
                'LOCATION_ORGANIZATION',
                'LOCATION_STREET',
                'LOCATION_ZIP',
                'NAME_DOCTOR',
                'NAME_OTHER',
                'NAME_PATIENT',
                'PROFESSION',
            ]
        source_dir = argv[1]
        pred_dir = argv[2]

        files = os.listdir(source_dir)
        ann_files = [f for f in files if f.endswith('.ann')]

        # process all files and collect results for each entity type
        scores = defaultdict(list)
        total_ref, total_pred = defaultdict(int), defaultdict(int)
        for f in ann_files:
            logging.debug(f'Processing {f}...')
            ref = read_ann(os.path.join(source_dir, f))
            pred = read_ann(os.path.join(pred_dir, f))

            for etype in etypes:
                eref = [r for r in ref if r[1] == etype]
                epred = [p for p in pred if p[1] == etype]
                scores[etype] += eval(eref, epred)
                total_ref[etype] += len(eref)
                total_pred[etype] += len(epred)

        # print results for each entity type
        for etype in etypes:
            print('Entity:', etype)

            # Lentient Precision, Recall and F1
            lenient = sum(scores[etype])
            precision, recall, f1 = prec_rec_f1(lenient, total_ref[etype], total_pred[etype])
            print(f'Lenient precision: {precision} ({lenient} / {total_pred[etype]})')
            print(f'Lenient recall: {recall} ({lenient} / {total_ref[etype]})')
            print(f'Lenient F1: {f1}')

            # Exact Precision, Recall and F1
            exact = len([s for s in scores[etype] if s == 1.0])
            precision, recall, f1 = prec_rec_f1(exact, total_ref[etype], total_pred[etype])
            print(f'Exact precision: {precision} ({exact} / {total_pred[etype]})')
            print(f'Exact recall: {recall} ({exact} / {total_ref[etype]})')
            print(f'Exact F1: {f1}')

            print()

        total_scores = sum(scores.values(), [])
        total_ref = sum(total_ref.values())
        total_pred = sum(total_pred.values())

        print(f'Global results (micro-averaged):')
        lenient = sum(total_scores)
        precision, recall, f1 = prec_rec_f1(lenient, total_ref, total_pred)
        print(f'Lenient precision: {precision} ({lenient} / {total_pred})')
        print(f'Lenient recall: {recall} ({lenient} / {total_ref})')
        print(f'Lenient F1: {f1}')
        exact = len([s for s in total_scores if s == 1.0])
        recision, recall, f1 = prec_rec_f1(exact, total_ref, total_pred)
        print(f'Exact precision: {precision} ({exact} / {total_pred})')
        print(f'Exact recall: {recall} ({exact} / {total_ref})')
        print(f'Exact F1: {f1}')
