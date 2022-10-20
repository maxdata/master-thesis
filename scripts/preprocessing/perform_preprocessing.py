from argparse import ArgumentParser
from typing import Optional

import wandb

from transformers import T5Tokenizer

from information_extraction.preprocessing.feature_extraction.text import TextExtractor
from information_extraction.preprocessing.feature_extraction.html import HtmlExtractor
from information_extraction.config import WANDB_PROJECT, DATA_DIR


def main(split: str, representation: str, parent_depth: Optional[int] = None, num_workers: int = 1):
    if split == 'random':
        slug = representation
    else:
        slug = f'{representation}-{split}'

    wandb.init(
        project=WANDB_PROJECT,
        name=f'{slug}-feature-extraction',
        group='feature-extraction',
        job_type='preprocess',
    )

    input_dir = DATA_DIR / f'SWDE-{split}-split'
    output_dir = DATA_DIR / f'SWDE-{slug}'

    input_artifact = wandb.use_artifact(f'{split}-split:latest')
    input_artifact.download(root=str(input_dir))

    tokenizer = T5Tokenizer.from_pretrained('t5-base')

    for context_size in [128, 256, 512]:
        if representation == 'html':
            # Use best performing configuration: encode id only, with split/exploded attribute values
            extractor = HtmlExtractor(input_dir, output_dir, tokenizer, context_size,
                                      parent_depth=parent_depth, encode_id=True, split_attributes=True,
                                      num_workers=num_workers)
            metadata = {'Parent depth': parent_depth}
        else:
            extractor = TextExtractor(input_dir, output_dir, tokenizer, context_size,
                                      num_workers=num_workers)
            metadata = None

        extractor.process_dataset()

    descriptions = {
        'text': 'Preprocessed SWDE dataset where contexts are represented as pure text',
        'html': 'Preprocessed SWDE dataset where HTML structure is included in the context',
    }

    artifact = wandb.Artifact(f'swde-{slug}', type='preprocessed-data', metadata=metadata,
                              description=descriptions[representation])
    artifact.add_dir(str(output_dir))

    wandb.log_artifact(artifact)
    wandb.finish()


if __name__ == '__main__':
    parser = ArgumentParser(description='Preprocess the SWDE dataset into segments and sequence representations')

    parser.add_argument('-s', '--split', choices=['random', 'webke', 'zero-shot'], default='random',
                        help='Which data split to use as input for the preprocessing')
    parser.add_argument('-r', '--representation', choices=['text', 'html'], required=True,
                        help='Which representation to use')
    parser.add_argument('-p', '--parent-depth', type=int,
                        help='The amount of ancestors to encode in the HTML representation')
    parser.add_argument('-n', '--num-workers', type=int, default=1,
                        help='The amount of worker processes to use for multi-processing')

    args = parser.parse_args()

    if args.representation == 'html' and args.parent_depth is None:
        raise ValueError('Parent depth must be supplied when representation is HTML')

    main(args.split, args.representation, args.parent_depth, args.num_workers)
