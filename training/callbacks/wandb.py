from typing import Dict, Optional

import wandb

from callbacks.base import BaseCallback


class WandbCallback(BaseCallback):
    def __init__(self, project_name: str, run_name: str, config: Optional[dict] = None, group: Optional[str] = None,
                 do_init: bool = True):
        self.project_name = project_name
        self.run_name = run_name
        self.config = config or {}
        self.group = group
        self.do_init = do_init

    def on_trainer_finish(self):
        if self.do_init:
            wandb.finish()

    def on_train_start(self, run_params: dict):
        if self.do_init:
            wandb.init(
                project=self.project_name,
                name=self.run_name,
                config={**run_params, **self.config},
                group=self.group,
                job_type='train',
            )

    def on_step_end(self, step_num: int, loss: float):
        wandb.log({'train/loss': loss}, step=step_num)

    def on_validation_end(self, step_num: int, metrics: Dict[str, float]):
        wandb.log({f'val/{key}': value for key, value in metrics.items()}, step=step_num)

    def on_evaluation_end(self, results: dict):
        summary_dict = {}
        tables = {}

        for split_name, split_data in results.items():
            for metric_name, metric_value in split_data['agg'].items():
                summary_dict[f'{split_name}/{metric_name}'] = metric_value

            if 'instances' in split_data:
                tables[f'instances/{split_name}'] = wandb.Table(dataframe=split_data['instances'])

            if 'documents' in split_data:
                tables[f'documents/{split_name}'] = wandb.Table(dataframe=split_data['documents'])

        if tables:
            wandb.log(tables)

        wandb.summary.update(summary_dict)
