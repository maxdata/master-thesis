from tqdm import tqdm
import wandb

from information_extraction.evaluation import Evaluator
from information_extraction.data.metrics import compute_f1, compute_exact

from information_extraction.training.callbacks import get_callbacks
from information_extraction.training.data import get_dataset, SWDEDataModule
from information_extraction.training.trainer import get_trainer, BaseTrainer


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
    wandb.init(job_type='train')

    config = wandb.config

    run_name = config.run_name.format(**dict(config))
    wandb.run.name = run_name

    dataset = get_dataset(config)
    trainer = get_trainer(config)
    trainer.callbacks.extend(get_callbacks(dataset, run_name, config))

    with trainer:
        trainer.train(dataset.train_dataloader(), config.num_steps, config.batch_size,
                      warmup_steps=config.get('warmup_steps', 0))

        perform_evaluation(trainer, dataset, config)

    wandb.finish()


if __name__ == '__main__':
    main()
