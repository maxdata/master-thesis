from typing import List
import uuid

import torch
from torch.nn.utils.rnn import pad_sequence
from sentence_transformers import SentenceTransformer

from information_extraction.dtypes import BertBatch
from information_extraction.training.data import BaseDataset
from information_extraction.data.metrics import normalize_answer
from information_extraction.config import DATA_DIR


class BertDataset(BaseDataset):
    word_indices: List[int]
    empty_embedding: torch.Tensor
    embedding_location: str

    def prepare_inputs(self):
        self.word_indices = []

        not_null_indices = []
        num_not_found = 0
        for index, (context, target) in enumerate(zip(self.inputs, self.targets)):
            normalized_target = normalize_answer(target)

            if not normalized_target:
                # Use -1 to indicate the value was not found
                self.word_indices.append(-1)
                continue

            for i, word in enumerate(context):
                if normalize_answer(word) == normalized_target:
                    self.word_indices.append(i)
                    break
            else:
                self.word_indices.append(-1)
                num_not_found += 1

        if self.remove_null:
            self.inputs = [self.inputs[i] for i in not_null_indices]
            self.targets = [self.targets[i] for i in not_null_indices]
            self.ancestors = [self.ancestors[i] for i in not_null_indices]
            self.features = [self.features[i] for i in not_null_indices]
            self.word_indices = [self.word_indices[i] for i in not_null_indices]
        elif num_not_found > 0:
            print(f'Warning: BertDataset found {num_not_found}/{len(self.inputs)} samples '
                  f'where the context does not contain the answer!')

        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        distinct_ancestors = list(set(x for a in self.ancestors for x in a))
        encoded_ancestors = torch.as_tensor(model.encode(distinct_ancestors, device=device, convert_to_numpy=True,
                                            show_progress_bar=True)).repeat(1, 2)

        embedding_mapping = {a: i for i, a in enumerate(distinct_ancestors)}
        self.ancestors = [
            [embedding_mapping[ancestors] if ancestors else None for ancestors in ancestors_per_word]
            for ancestors_per_word in self.ancestors
        ]
        self.empty_embedding = torch.zeros_like(encoded_ancestors[0])

        self.embedding_location = str(DATA_DIR / f'{uuid.uuid4().hex}.pickle')
        torch.save(encoded_ancestors, self.embedding_location)

    def __getitem__(self, idx: List[int]) -> BertBatch:
        docs = [self.docs[i] for i in idx]
        inputs = [self.inputs[i] for i in idx]
        ancestors = [self.ancestors[i] for i in idx]
        targets = [self.targets[i] for i in idx]
        features = [self.features[i] for i in idx]
        word_indices = [self.word_indices[i] for i in idx]

        encoding = self.tokenizer([[f] for f in features], inputs, **self.tokenize_kwargs)

        start_positions = []
        end_positions = []

        for batch_index, word_index in enumerate(word_indices):
            if word_index < 0:
                # In this case, the answer does not exist in the context
                start_positions.append(0)
                end_positions.append(0)
            else:
                token_span = encoding.word_to_tokens(batch_index, word_index, sequence_index=1)

                if token_span is not None:
                    start_positions.append(token_span.start)
                    end_positions.append(token_span.end - 1)
                else:
                    start_positions.append(0)
                    end_positions.append(0)

        embedding_matrix = torch.load(self.embedding_location)
        html_embeddings = []
        for batch_index, ancestors_per_word in enumerate(ancestors):
            current_embeddings = []

            for token_index in range(encoding.input_ids.shape[1]):
                word_index = encoding.token_to_word(batch_index, token_index)

                if encoding.token_to_sequence(batch_index, word_index) == 1 and word_index is not None:
                    current_embeddings.append(
                        embedding_matrix[ancestors_per_word[word_index]]
                        if ancestors_per_word[word_index] is not None
                        else self.empty_embedding
                    )
                else:
                    current_embeddings.append(self.empty_embedding)

            html_embeddings.append(torch.stack(current_embeddings))

        html_embeddings = pad_sequence(html_embeddings, batch_first=True)

        return BertBatch(
            docs,
            inputs,
            html_embeddings,
            targets,
            features,
            encoding.input_ids,
            encoding.attention_mask,
            encoding.token_type_ids,
            torch.as_tensor(start_positions),
            torch.as_tensor(end_positions),
        )
