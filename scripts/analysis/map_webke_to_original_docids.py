from collections import defaultdict
import json
import os
from pprint import pprint

from information_extraction.config import DATA_DIR
from information_extraction.analysis.tables import get_wandb_artifact


EXPANDED_SWDE_DIR = DATA_DIR / 'ExpandedSWDE'
TARGET_FILENAME = 'webke_split_mapping.json'


def flatten_values(data):
    return set([
        (formatted_value, value)
        for key, values in data.items()
        for value in values
        if (formatted_value := key.split('|')[-1].split('&&&')[0].strip()) != ''
        and key != 'topic_entity_name'
    ])


def read_ground_truths():
    ground_truths = defaultdict(dict)

    for vertical in os.listdir(EXPANDED_SWDE_DIR):
        for filename in os.listdir(EXPANDED_SWDE_DIR / vertical):
            # website = filename.split('-')[1].split('(')[0]
            website = filename.split('.')[0]
            with open(EXPANDED_SWDE_DIR / vertical / filename) as _file:
                data = json.load(_file)

            ground_truths[vertical][website] = {
                key[:-4]: flatten_values(values) for key, values in data.items()
            }

    return ground_truths


def read_webke_data():
    webke_dir = get_wandb_artifact('webke-predictions')

    data = defaultdict(lambda: defaultdict(dict))

    for vertical in os.listdir(webke_dir):
        for split_name in os.listdir(webke_dir / vertical):
            for filename in os.listdir(webke_dir / vertical / split_name):
                doc_id = filename.split('.')[0]

                with open(webke_dir / vertical / split_name / filename) as _file:
                    file_data = json.load(_file)

                data[vertical][split_name][doc_id] = set(tuple(entry) for entry in file_data['pred_list'])

    return data


def find_ground_truth_match(data, ground_truths):
    # for vertical, websites in ground_truths.items():
    for website, files in ground_truths.items():
        for doc_id, file_data in files.items():
            if len(file_data) == 0 and len(data) == 0:
                yield website, doc_id
            elif len(data) > 0 and data.issubset(file_data):
                yield website, doc_id


def generate_mapping(ground_truths, webke_data):
    results = defaultdict(lambda: defaultdict(dict))
    not_mapped = dict()

    for vertical in ['movie', 'nbaplayer', 'university']:
        for split_name, split_data in webke_data[vertical].items():
            for doc_id, data in split_data.items():
                possible_ground_truths = list(find_ground_truth_match(data, ground_truths[vertical]))

                if len(possible_ground_truths) != 1:
                    not_mapped[(split_name, vertical, doc_id)] = possible_ground_truths
                    print(f'{len(possible_ground_truths)} found for {vertical}/{doc_id}')
                    print(data)
                else:
                    results[split_name][vertical][doc_id] = possible_ground_truths[0]

    pprint(dict(results))

    mapped_files = {
        (split_name, vertical): set(vertical_data.values())
        for split_name, split_data in results.items()
        for vertical, vertical_data in split_data.items()
    }

    for (split_name, vertical, doc_id), possible_ground_truths in dict(not_mapped).items():
        remaining_ground_truths = set(possible_ground_truths) - mapped_files[(split_name, vertical)]

        if len(remaining_ground_truths) == 1:
            results[split_name][vertical][doc_id] = list(remaining_ground_truths)[0]
            not_mapped.pop((split_name, vertical, doc_id))

    print('No mapping found for', len(not_mapped), 'files:', list(not_mapped))

    missing_train = len([x for x, _, _ in not_mapped if x == 'train'])
    missing_test = len([x for x, _, _ in not_mapped if x == 'test'])

    found_train = len([y for x in results['train'].values() for y in x.values()])
    found_test = len([y for x in results['test'].values() for y in x.values()])

    print(f'Train: not found for {missing_train}/{missing_train+found_train} = '
          f'{missing_train/(missing_train+found_train) * 100:.4f}%')
    print(f'Test: not found for {missing_test}/{missing_test + found_test} = '
          f'{missing_test / (missing_test + found_test) * 100:.4f}%')

    return results


def main():
    if os.path.exists(TARGET_FILENAME):
        print('Mapping already exists!')
        return

    ground_truths = read_ground_truths()
    webke_data = read_webke_data()

    mapping = generate_mapping(ground_truths, webke_data)

    with open(TARGET_FILENAME, 'w') as _file:
        json.dump(mapping, _file)


if __name__ == '__main__':
    main()
