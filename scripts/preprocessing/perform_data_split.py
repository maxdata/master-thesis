from argparse import ArgumentParser
from pathlib import Path
import shutil
import tempfile

from information_extraction.preprocessing.data_split import get_random_split, get_webke_split, get_zero_shot_split
from information_extraction.analysis.mappings import WEBKE_SPLIT_MAPPING
from information_extraction.config import WANDB_PROJECT, DATA_DIR

import wandb


def normalize_name(name: str):
    """Removes the vertical indicator and the amount of pages in the website"""
    return name.split('(')[0].split('-')[1]


def main(input_artifact: str, split_mode: str, train_size: float):
    with tempfile.TemporaryDirectory() as tmpdir:
        wandb.init(
            project=WANDB_PROJECT,
            name=f'{split_mode}-split',
            job_type='preprocess',
        )

        input_dir = DATA_DIR / input_artifact
        output_dir = Path(tmpdir)

        dataset_artifact = wandb.use_artifact(f'{input_artifact}:latest')
        dataset_artifact.download(root=str(input_dir))

        if split_mode == 'random':
            split = get_random_split(input_dir, train_size)
        elif split_mode == 'webke':
            split = get_webke_split(WEBKE_SPLIT_MAPPING, train_size)
        elif split_mode == 'zero-shot':
            split = get_zero_shot_split(input_dir, train_size)
        else:
            raise ValueError(f'Split mode `{split_mode}` not supported')

        for split_name, split_files in split.items():
            for vertical, website, filename in split_files:
                dest_path = output_dir / split_name / vertical / normalize_name(website)

                dest_path.mkdir(parents=True, exist_ok=True)

                shutil.copy(
                    input_dir / vertical / website / filename,
                    dest_path / filename,
                )

        shutil.copytree(input_dir / 'groundtruth', output_dir / 'groundtruth')

        descriptions = {
            'random': 'Fully random train / val / test split of the SWDE dataset',
            'webke': 'Train / val / test split of the SWDE dataset matching the preprocessed WebKE data',
            'zero-shot': 'Zero shot train / val / test split of the SWDE dataset',
        }

        val_test_size = round((1 - train_size) / 2, 2)
        metadata = {
            'train_size': train_size,
            'val_size': val_test_size,
            'test_size': val_test_size,
        }

        artifact = wandb.Artifact(f'{split_mode}-split', type='train-val-test-split', metadata=metadata,
                                  description=descriptions[split_mode])
        artifact.add_dir(str(output_dir))

        wandb.log_artifact(artifact)
        wandb.finish()


if __name__ == '__main__':
    parser = ArgumentParser(description='Perform a train/validation/test data split on the SWDE dataset')

    parser.add_argument('-i', '--input-artifact', choices=['swde', 'swde-pos'], default='swde',
                        help='The dataset to use as input for the data split')
    parser.add_argument('-s', '--split-mode', choices=['random', 'webke', 'zero-shot'], default='random',
                        help='Which type of data split to use')
    parser.add_argument('-t', '--train-size', type=float, required=True,
                        help='The size of the training set')

    args = parser.parse_args()

    main(args.input_artifact, args.split_mode, args.train_size)
