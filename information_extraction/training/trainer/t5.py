import math
from typing import Optional, Tuple

from transformers import T5Tokenizer, T5ForConditionalGeneration, LogitsProcessor, LogitsProcessorList

import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from . import BaseTrainer
from information_extraction.dtypes import SegmentPrediction, T5Batch


class T5Trainer(BaseTrainer):
    architecture = 't5'

    def __init__(
        self,
        model_version: str,
        *args,
        **kwargs,
    ):
        model = T5ForConditionalGeneration.from_pretrained(model_version)
        tokenizer = T5Tokenizer.from_pretrained(model_version)

        self.loss_fn = CrossEntropyLoss()

        super().__init__(
            model,
            tokenizer,
            *args,
            **kwargs
        )

    def forward(self, batch: T5Batch):
        return self.model(
            input_ids=batch.input_ids.to(self.device),
            attention_mask=batch.attention_mask.to(self.device),
            decoder_input_ids=batch.decoder_input_ids.to(self.device),
            decoder_attention_mask=batch.decoder_attention_mask.to(self.device),
        )

    def train_step(self, batch: T5Batch) -> torch.Tensor:
        prediction = self.forward(batch)

        logits = prediction.logits
        target_labels = batch.target_labels.to(self.device)
        return self.loss_fn(logits.view(-1, logits.size(-1)), target_labels.flatten())

    def predict_segment_batch(self, batch: T5Batch) -> Tuple[float, SegmentPrediction]:
        with torch.no_grad():
            loss = float(self.train_step(batch)) if any(batch.targets) else 0
            num_beams = self.model_kwargs.get('num_beams')
            logits_processor = CopyInputLogitsProcessor(batch.input_ids.to(self.device), self.tokenizer.eos_token_id,
                                                        num_beams=num_beams)
            logits_processor_list = LogitsProcessorList([logits_processor])

            outputs = self.model.generate(batch.input_ids.to(self.device),
                                          num_beams=num_beams,
                                          logits_processor=logits_processor_list,
                                          return_dict_in_generate=True, output_scores=True)

            if num_beams is None:
                # Compute the probability of each token in the decoded sequences
                stacked_scores = F.softmax(torch.stack(outputs.scores).permute(1, 0, 2), dim=2).double()
                token_scores = torch.take_along_dim(stacked_scores, outputs.sequences[:, 1:, None], dim=2).squeeze()

                # Average the probabilities over each sequence to obtain score per sequence
                token_mask = outputs.sequences[:, 1:].ne(self.tokenizer.pad_token_id)
                sums = torch.where(token_mask, token_scores, 0.).sum(dim=1)
                lengths = token_mask.sum(dim=1)

                scores = torch.nan_to_num(sums / lengths).tolist()
            else:
                # Use the beam score as confidence measure
                scores = torch.exp(outputs.sequences_scores).tolist()

        predictions = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)

        return loss, SegmentPrediction(batch, predictions, scores)


class CopyInputLogitsProcessor(LogitsProcessor):
    r"""
    :class:`transformers.LogitsProcessor` that constrains generation to sequences that exist in the input sequence
    Args:
        original_input_ids: (:obj:`torch.Tensor`):
            The tokenized input sequence for this batch, from which the generation must copy.
        eos_token: (:obj:`int`):
            The token ID of the EOS token, to ensure this is not masked away.
    """

    def __init__(self, original_input_ids: torch.Tensor, eos_token: int, num_beams: Optional[int] = None):
        self.original_input_ids = original_input_ids
        self.eos_token = eos_token
        self.num_beams = num_beams

        if num_beams is not None:
            self.original_input_ids = torch.repeat_interleave(original_input_ids, num_beams, dim=0)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, -math.inf)
        mask[:, self.eos_token] = 0

        # Strip the starting pad token
        input_ids = input_ids[:, 1:]

        possible_sequences = self.original_input_ids.unfold(1, input_ids.shape[1] + 1, 1)
        allowed_sequences = (possible_sequences[:, :, :input_ids.shape[1]] == input_ids[:, None, :]).all(axis=-1)

        for batch_id in range(allowed_sequences.shape[0]):
            allowed_tokens = possible_sequences[batch_id][allowed_sequences[batch_id]][:, -1]
            mask[batch_id, allowed_tokens] = 0

        return scores + mask
