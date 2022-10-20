from typing import List, Tuple

from transformers import BertTokenizer, BertForQuestionAnswering
from transformers.modeling_outputs import QuestionAnsweringModelOutput

import torch
from torch.nn import functional as F

from . import BaseTrainer
from information_extraction.dtypes import BertBatch, SegmentPrediction


class BertTrainer(BaseTrainer):
    architecture = 'bert'

    def __init__(
        self,
        model_version: str,
        *args,
        **kwargs,
    ):
        model = BertForQuestionAnswering.from_pretrained(model_version)
        tokenizer = BertTokenizer.from_pretrained(model_version)

        super().__init__(
            model,
            tokenizer,
            *args,
            **kwargs
        )

    def forward(self, batch: BertBatch):
        input_embeddings = self.model.bert.get_input_embeddings()(batch.input_ids.to(self.device))
        input_embeddings += batch.html_embeddings.to(self.device)

        return self.model(
            inputs_embeds=input_embeddings,
            attention_mask=batch.attention_mask.to(self.device),
            token_type_ids=batch.token_type_ids.to(self.device),
            start_positions=batch.start_positions.to(self.device),
            end_positions=batch.end_positions.to(self.device),
        )

    def train_step(self, batch: BertBatch) -> torch.Tensor:
        prediction = self.forward(batch)

        return prediction.loss

    def predict_segment_batch(self, batch: BertBatch) -> Tuple[float, SegmentPrediction]:
        with torch.no_grad():
            outputs = self.forward(batch)

            scores, spans = self.extract_spans(batch, outputs)

        predictions = self.spans_to_predictions(batch, spans)

        return float(outputs.loss), SegmentPrediction(batch, predictions, scores)

    def spans_to_predictions(self, batch: BertBatch, spans: torch.Tensor):
        predictions = []

        for i, (start, end) in enumerate(spans):
            if start == end == 0:
                predictions.append('')
            else:
                predict_answer_tokens = batch.input_ids[i, start:end + 1]
                predictions.append(self.tokenizer.decode(predict_answer_tokens, skip_special_tokens=True))

        return predictions

    @staticmethod
    def extract_spans(batch: BertBatch, outputs: QuestionAnsweringModelOutput) -> Tuple[List[float], torch.Tensor]:
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        # Ensure output span can only be found in the context or the CLS token
        mask = batch.token_type_ids == 0
        mask[:, 0] = False

        start_logits[mask] = -10000
        end_logits[mask] = -10000

        # Obtain the matrix of possible probabilities
        start_probs = F.softmax(start_logits, dim=-1)
        end_probs = F.softmax(end_logits, dim=-1)
        scores = torch.triu(start_probs[:, :, None] * end_probs[:, None, :])

        # Ensure it cannot predict a non-empty span starting with the CLS token
        scores[:, 0, 1:] = 0
        scores[:, 1:, 0] = 0

        max_scores, indices = torch.max(scores.view(scores.shape[0], -1), dim=-1)
        indices = torch.stack([torch.div(indices, scores.shape[-1], rounding_mode='floor'),
                               indices % scores.shape[-1]], dim=1)

        return max_scores.tolist(), indices
