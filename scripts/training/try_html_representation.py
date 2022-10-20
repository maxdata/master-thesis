from collections import namedtuple
import os
from pathlib import Path
import tempfile

import pandas as pd
from tqdm import tqdm
import wandb

from transformers import T5Tokenizer

from information_extraction.evaluation import Evaluator
from information_extraction.data.metrics import compute_f1, compute_exact

from information_extraction.training.callbacks import get_callbacks
from information_extraction.training.data import SWDEDataModule
from information_extraction.training.trainer import get_trainer, BaseTrainer

from information_extraction.preprocessing.feature_extraction.text import TextExtractor
from information_extraction.preprocessing.feature_extraction.html import HtmlExtractor, SimpleHtmlExtractor


Config = namedtuple('Config', ['encode_id', 'encode_class', 'encode_tag_subset', 'split_attributes'])

GENERAL_TAGS = {'div', 'span', 'a', 'p'}
NUM_PREPROCESS_WORKERS = 48


configs = {
    'html-base': Config(False, False, None, False),
    'html-id': Config(True, False, None, False),
    'html-class': Config(False, True, None, False),
    'html-id-class': Config(True, True, None, False),
    'html-id-class-subset': Config(True, True, GENERAL_TAGS, False),
    'html-id-expanded': Config(True, False, None, True),
    'html-class-expanded': Config(False, True, None, True),
    'html-id-class-expanded': Config(True, True, None, True),
    'html-id-class-expanded-subset': Config(True, True, GENERAL_TAGS, True),
}


def get_dataset(config: wandb.Config) -> SWDEDataModule:
    data_path = Path('~/Data/SWDE-split').expanduser()
    input_artifact = wandb.use_artifact('random-split:latest')
    input_artifact.download(str(data_path))

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir)

        tokenizer = T5Tokenizer.from_pretrained('t5-base')

        if config.representation == 'text':
            extractor = TextExtractor(data_path, output_path, tokenizer,
                                      max_length=config.context_size, num_workers=NUM_PREPROCESS_WORKERS)
        elif config.representation == 'html-simple':
            extractor = SimpleHtmlExtractor(data_path, output_path, tokenizer,
                                            max_length=config.context_size, num_workers=NUM_PREPROCESS_WORKERS)
        else:
            extractor_config = configs[config.representation]
            extractor = HtmlExtractor(data_path, output_path, tokenizer,
                                      parent_depth=config.parent_depth,
                                      encode_id=extractor_config.encode_id,
                                      encode_class=extractor_config.encode_class,
                                      encode_tag_subset=extractor_config.encode_tag_subset,
                                      split_attributes=extractor_config.split_attributes,
                                      max_length=config.context_size, num_workers=NUM_PREPROCESS_WORKERS)

        dfs = {
            split: pd.concat([
                pd.read_csv(extractor.process_directory(split, config.vertical, website))
                for website in os.listdir(data_path / split / config.vertical)
            ])
            for split in ['train', 'val', 'test']
        }

        for split, split_df in dfs.items():
            split_df.to_csv(output_path / f'{split}.csv', index=False)

        models = {
            'bert': {
                'model_version': 'SpanBERT/spanbert-base-cased',
                'mini_batch_size': 16 if config.context_size == 512 else 64,
            },
            't5': {
                'model_version': 't5-base',
                'mini_batch_size': 8 if config.context_size == 512 else 32,
            }
        }

        model_config = models[config.model]

        dataset = SWDEDataModule(model_config['model_version'],
                                 max_length=config.context_size,
                                 train_files=[output_path / 'train.csv'],
                                 val_files=[output_path / 'val.csv'],
                                 test_files=[output_path / 'test.csv'],
                                 batch_size=model_config['mini_batch_size'],
                                 num_workers=config.num_workers,
                                 remove_null=config.remove_null)

    return dataset


def perform_evaluation(trainer: BaseTrainer, dataset: SWDEDataModule, config: wandb.Config):
    eval_loaders = {
        'train': lambda: dataset.train_document_dataloader(num_documents=len(set(dataset.data_val.docs))),
        'val': lambda: dataset.val_document_dataloader(),
        'test': lambda: dataset.test_document_dataloader(),
    }

    evaluator = Evaluator(metrics={'f1': compute_f1, 'em': compute_exact})

    for split in config.get('evaluation_datasets', ['train', 'val', 'test']):
        dataloader = tqdm(eval_loaders[split](), desc=f'Evaluating {split}')
        predictions = trainer.predict_documents(dataloader, method=config.get('evaluation_method', 'greedy'))
        results = evaluator.evaluate_documents(predictions)

        for callback in trainer.callbacks:
            callback.on_evaluation_end(split, results)


def main():
    wandb.init(job_type='experiment')

    config = wandb.config

    run_name = config.run_name.format(**dict(config))
    wandb.run.name = run_name

    trainer = get_trainer(config)
    dataset = get_dataset(config)
    trainer.callbacks.extend(get_callbacks(dataset, run_name, config))

    with trainer:
        trainer.train(dataset.train_dataloader(), config.num_steps, config.batch_size,
                      warmup_steps=config.get('warmup_steps', 0))

        perform_evaluation(trainer, dataset, config)

    wandb.finish()


if __name__ == '__main__':
    main()
