from pathlib import Path

import pandas as pd
import wandb

from callbacks import EarlyStopping, ModelCheckpoint, WandbCallback
from data.module import SWDEDataModule
from metrics import compute_f1, compute_exact, f1_metric, em_metric
from trainer import BaseTrainer, BertTrainer, T5Trainer


MODELS = {
    'bert': {
        'model_version': 'SpanBERT/spanbert-base-cased',
        'trainer_class': BertTrainer,
    },
    't5': {
        'model_version': 't5-base',
        'trainer_class': T5Trainer,
    }
}


def get_dataset(config: wandb.Config) -> SWDEDataModule:
    if ':' in config.representation:
        representation, tag = config.representation.split(':')
    else:
        representation = config.representation
        tag = 'latest'

    data_path = Path(f'~/Data/SWDE-{representation}-{tag}').expanduser()
    input_artifact = wandb.use_artifact(f'swde-{representation}:{tag}')
    input_artifact.download(str(data_path))

    train_files = list(data_path.glob(f'train/{config.vertical}/*/*-{config.context_size}.csv'))
    val_files = list(data_path.glob(f'val/{config.vertical}/*/*-{config.context_size}.csv'))
    test_files = list(data_path.glob(f'test/{config.vertical}/*/*-{config.context_size}.csv'))

    print('Files used for training:', len([str(f) for f in train_files]))
    print('Files used for validation:', len([str(f) for f in val_files]))
    print('Files used for testing:', len([str(f) for f in test_files]))

    mini_batch_sizes = {
        'bert': 16 if config.context_size == 512 else 64,
        't5': 8 if config.context_size == 512 else 32,
    }

    dataset = SWDEDataModule(MODELS[config.model]['model_version'],
                             max_length=config.context_size,
                             train_files=train_files,
                             val_files=val_files,
                             test_files=test_files,
                             batch_size=mini_batch_sizes[config.model],
                             num_workers=config.num_workers,
                             remove_null=config.remove_null)

    return dataset


def get_trainer(dataset: SWDEDataModule, run_name: str, config: wandb.Config):
    monitor_mode = 'min' if config.monitor_metric == 'loss' else 'max'

    callbacks = [
        ModelCheckpoint(f'models/{run_name}.state_dict', restore=True,
                        metric=config.monitor_metric, mode=monitor_mode),
        WandbCallback('information_extraction', run_name, do_init=False),
        EarlyStopping(patience=config.early_stopping_patience,
                      metric=config.monitor_metric, mode=monitor_mode),
    ]

    trainer_kwargs = {
        'train_loader': dataset.train_dataloader(),
        'val_loader': dataset.val_dataloader(size=config.validation_batches * config.batch_size),
        'validate_per_steps': config.validation_interval,
        'metrics': {
            'f1': compute_f1,
            'em': compute_exact,
        },
        'learning_rate': config.learning_rate,
        'optimizer': config.optimizer,
        'callbacks': callbacks,
    }

    model_config = MODELS[config.model]

    return model_config['trainer_class'](model_config['model_version'],
                                         **trainer_kwargs)


def main():
    wandb.init(job_type='train')

    config = wandb.config

    run_name = config.run_name.format(**dict(config))
    wandb.run.name = run_name

    dataset = get_dataset(config)

    with get_trainer(dataset, run_name, config) as trainer:
        trainer.train(config.num_steps, config.batch_size)

        eval_loaders = {
            'train': lambda: dataset.train_dataloader(size=len(dataset.data_val)),
            'val': lambda: dataset.val_dataloader(),
            'test': lambda: dataset.test_dataloader(),
        }

        trainer.perform_evaluation(**{
            dataset: eval_loaders[dataset]()
            for dataset in config.get('evaluation_datasets',
                                      ['train', 'val', 'test'])
        })

    wandb.finish()


if __name__ == '__main__':
    main()
