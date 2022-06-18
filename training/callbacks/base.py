from outputs import EvaluationResult


class BaseCallback:
    trainer = None

    def on_trainer_init(self, trainer):
        self.trainer = trainer

    def on_trainer_finish(self):
        pass

    def on_train_start(self, run_params: dict):
        pass

    def on_train_end(self):
        pass

    def on_step_end(self, step_num: int, loss: float):
        pass

    def on_validation_end(self, step_num: int, results: EvaluationResult):
        pass

    def on_evaluation_end(self, identifier: str, results: EvaluationResult):
        pass
