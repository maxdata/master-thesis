from pathlib import Path

import wandb

from callbacks import EarlyStopping, ModelCheckpoint, SavePredictions, ValidationCallback, WandbCallback
from data.ground_truths import GroundTruths
from data.module import SWDEDataModule
from evaluation import Evaluator
from metrics import compute_f1, compute_exact
from trainer import BertTrainer, T5Trainer


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

    website = config.get('website', '*')
    train_files = list(data_path.glob(f'train/{config.vertical}/{website}/*-{config.context_size}.csv'))
    val_files = list(data_path.glob(f'val/{config.vertical}/{website}/*-{config.context_size}.csv'))
    test_files = list(data_path.glob(f'test/{config.vertical}/{website}/*-{config.context_size}.csv'))

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
        SavePredictions(documents_dir=f'results/{run_name}', segments_dir=f'results/{run_name}'),
    ]

    trainer_kwargs = {
        'train_loader': dataset.train_dataloader(),
        'learning_rate': config.learning_rate,
        'optimizer': config.optimizer,
        'callbacks': callbacks,
        'num_beams': config.get('num_beams'),
    }

    model_config = MODELS[config.model]

    trainer = model_config['trainer_class'](model_config['model_version'],
                                            **trainer_kwargs)

    ground_truths = GroundTruths(Path('~/Data/SWDE/groundtruth/').expanduser())
    evaluator = Evaluator(trainer, metrics={'f1': compute_f1, 'em': compute_exact}, ground_truths=ground_truths)

    if 'validation_documents' in config:
        val_callback = ValidationCallback(evaluator, config.validation_interval,
                                          document_loader=dataset.val_document_dataloader(config.validation_documents),
                                          label='Validating')
        trainer.callbacks.append(val_callback)
    elif 'validation_batches' in config:
        val_callback = ValidationCallback(evaluator, config.validation_interval,
                                          segment_loader=dataset.val_dataloader(
                                              size=config.validation_batches * config.batch_size
                                          ),
                                          label='Validating')
        trainer.callbacks.append(val_callback)

    return trainer


def main():
    wandb.init(job_type='train')

    config = wandb.config

    run_name = config.run_name.format(**dict(config))
    wandb.run.name = run_name

    dataset = get_dataset(config)

    with get_trainer(dataset, run_name, config) as trainer:
        trainer.train(config.num_steps, config.batch_size)

        eval_loaders = {
            'train': lambda: dataset.train_document_dataloader(num_documents=10000),
            'val': lambda: dataset.val_document_dataloader(),
            'test': lambda: dataset.test_document_dataloader(),
        }

        ground_truths = GroundTruths(Path('~/Data/SWDE/groundtruth/').expanduser())
        evaluator = Evaluator(trainer, metrics={'f1': compute_f1, 'em': compute_exact}, ground_truths=ground_truths)

        for split in config.get('evaluation_datasets', ['train', 'val', 'test']):
            results = evaluator.evaluate_documents(eval_loaders[split](), label=f'Evaluating {split}')

            for callback in trainer.callbacks:
                callback.on_evaluation_end(split, results)

    wandb.finish()


if __name__ == '__main__':
    main()
