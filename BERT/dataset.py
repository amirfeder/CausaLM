from random import random
from torch.utils.data import TensorDataset, Dataset
from tqdm import tqdm
from transformers.tokenization_bert import BertTokenizer
from BERT.lm_finetuning.MLM.pregenerate_training_data import CLS_TOKEN, SEP_TOKEN
from constants import BERT_PRETRAINED_MODEL, MAX_SENTIMENT_SEQ_LENGTH
from ast import literal_eval
from abc import abstractmethod
from typing import List
import pandas as pd
import numpy as np
import torch


class InputExample:
    def __init__(self, unique_id, text, label):
        self.unique_id = unique_id
        self.text = text
        self.label = label


class InputFeatures:
    def __init__(self, unique_id, tokens, input_ids, input_mask):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask


class InputLabel:
    def __init__(self, unique_id, label):
        self.unique_id = unique_id
        self.label = label


class BertTextDataset(Dataset):

    PAD_TOKEN_IDX = 0

    def __init__(self, data_path: str, treatment: str, subset: str, text_column: str, label_column: str,
                 bert_pretrained_model: str = BERT_PRETRAINED_MODEL, max_seq_length: int = MAX_SENTIMENT_SEQ_LENGTH):
        super().__init__()
        if subset not in ("train", "dev", "test", "train_debug", "dev_debug", "test_debug"):
            raise ValueError("subset argument must be {train, dev,test}")
        self.dataset_file = f"{data_path}/{treatment}_{subset}.csv"
        self.subset = subset
        self.text_column = text_column
        self.label_column = label_column
        self.max_seq_length = max_seq_length
        self.tokenizer = BertTokenizer.from_pretrained(bert_pretrained_model,
                                                       do_lower_case=bool(BERT_PRETRAINED_MODEL.endswith("uncased")))
        self.dataset = self.preprocessing_pipeline()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def preprocessing_pipeline(self):
        examples = self.read_examples()
        features, labels = self.convert_examples_to_features(examples)
        dataset = self.create_tensor_dataset(features, labels)
        return dataset

    def read_examples(self) -> List[InputExample]:
        """Read a list of `InputExample`s from an input file."""
        df = pd.read_csv(self.dataset_file, header=0, encoding='utf-8')
        return df.apply(self.read_examples_func, axis=1).tolist()

    @abstractmethod
    def read_examples_func(self, row: pd.Series) -> InputExample: ...

    @abstractmethod
    def convert_examples_to_features(self, examples: List[InputExample]) -> (List[InputFeatures], List[InputLabel]): ...

    @staticmethod
    def create_tensor_dataset(features: List[InputFeatures], labels: List[InputLabel]) -> TensorDataset:
        input_ids_list = list()
        input_masks_list = list()
        input_unique_id_list = list()
        input_labels_list = list()
        for f, l in zip(features, labels):
            input_ids_list.append(f.input_ids)
            input_masks_list.append(f.input_mask)
            assert l.unique_id == f.unique_id
            input_unique_id_list.append(f.unique_id)
            input_labels_list.append(l.label)
        all_input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        all_input_mask = torch.tensor(input_masks_list, dtype=torch.long)
        all_labels = torch.tensor(input_labels_list, dtype=torch.long)
        all_unique_id = torch.tensor(input_unique_id_list, dtype=torch.long)

        return TensorDataset(all_input_ids, all_input_mask, all_labels, all_unique_id)


class BertTextClassificationDataset(BertTextDataset):

    IGNORE_LABEL_IDX = -1

    def __init__(self, data_path: str, treatment: str, subset: str, text_column: str, label_column: str,
                 bert_pretrained_model: str = BERT_PRETRAINED_MODEL, max_seq_length: int = MAX_SENTIMENT_SEQ_LENGTH):
        super().__init__(data_path, treatment, subset, text_column, label_column, bert_pretrained_model, max_seq_length)

    def read_examples_func(self, row):
        return InputExample(unique_id=int(row.iloc[0]), text=str(row[self.text_column]), label=int(row[self.label_column]))

    def convert_examples_to_features(self, examples):
        """Loads a data file into a list of `InputFeature`s."""
        features_list = list()
        labels_list = list()
        for i, example in tqdm(enumerate(examples), total=len(examples), desc=f"{self.subset}-convert_examples_to_features"):
            features, example_len = self.tokenize_and_pad_sequence(example)
            features_list.append(features)
            labels_list.append(InputLabel(unique_id=example.unique_id, label=example.label))
        return features_list, labels_list

    def tokenize_and_pad_sequence(self, example):
        tokens = self.tokenizer.tokenize(example.text)

        tokens = tuple([CLS_TOKEN] + truncate_seq_first(tokens, self.max_seq_length) + [SEP_TOKEN])

        example_len = len(tokens) - 2

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < self.max_seq_length:
            input_ids.append(self.PAD_TOKEN_IDX)
            input_mask.append(self.PAD_TOKEN_IDX)

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length

        return InputFeatures(unique_id=example.unique_id, tokens=tokens,
                             input_ids=input_ids, input_mask=input_mask), example_len


class BertTokenClassificationDataset(BertTextDataset):

    IGNORE_LABEL_IDX = -100

    def __init__(self, data_path: str, treatment: str, subset: str, text_column: str, label_column: str,
                 bert_pretrained_model: str = BERT_PRETRAINED_MODEL, max_seq_length: int = MAX_SENTIMENT_SEQ_LENGTH):
        super().__init__(data_path, treatment, subset, text_column, label_column, bert_pretrained_model, max_seq_length)

    def read_examples_func(self, row):
        unique_id = int(row.iloc[0])
        text = str(row[self.text_column])
        labels_list = [int(i) for i in literal_eval(str(row[self.label_column]))]
        return InputExample(unique_id=unique_id, text=text, label=labels_list)

    def convert_examples_to_features(self, examples):
        """Loads a data file into a list of `InputFeature`s."""
        features_list = list()
        labels_list = list()
        # seq_lengths = list()
        for i, example in tqdm(enumerate(examples), total=len(examples), desc=f"{self.subset}-convert_examples_to_features"):
            features, example_len, labels = self.tokenize_and_pad_sequence(example)
            features_list.append(features)
            labels_list.append(labels)
            # seq_lengths.append(example_len)
        # print_seq_lengths_stats(None, seq_lengths, self.max_seq_length)
        return features_list, labels_list

    def tokenize_and_pad_sequence(self, example):
        tokens = self.tokenizer.tokenize(example.text)

        tokens = tuple([CLS_TOKEN] + truncate_seq_first(tokens, self.max_seq_length) + [SEP_TOKEN])
        labels = [self.IGNORE_LABEL_IDX] + truncate_seq_first(example.label, self.max_seq_length) + [self.IGNORE_LABEL_IDX]

        example_len = len(tokens) - 2

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < self.max_seq_length:
            input_ids.append(self.PAD_TOKEN_IDX)
            input_mask.append(self.PAD_TOKEN_IDX)
            labels.append(self.IGNORE_LABEL_IDX)

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(labels) == self.max_seq_length

        return InputFeatures(unique_id=example.unique_id, tokens=tokens,
                             input_ids=input_ids, input_mask=input_mask),\
               example_len,\
               InputLabel(unique_id=example.unique_id, label=labels)


def print_seq_lengths_stats(logger, text_seq_lengths, max_seq_length):
    logger.info(f"Num Sequences: {len(text_seq_lengths)}")
    logger.info(f"Minimum Sequence Length: {np.min(text_seq_lengths)}")
    logger.info(f"Average Sequence Length: {np.mean(text_seq_lengths)}")
    logger.info(f"Median Sequence Length: {np.median(text_seq_lengths)}")
    logger.info(f"99th Percentile Sequence Length: {np.percentile(text_seq_lengths, 99)}")
    logger.info(f"Maximum Sequence Length: {np.max(text_seq_lengths)}")
    logger.info(f"Num of over Maximum Sequence Length: {len([i for i in text_seq_lengths if i >= max_seq_length])}")


def truncate_seq_random_sub(tokens, max_seq_length):
    max_num_tokens = max_seq_length - 2
    l = 0
    r = len(tokens)
    trunc_tokens = list(tokens)
    while r - l > max_num_tokens:
        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if random() < 0.5:
            l += 1
        else:
            r -= 1
    return trunc_tokens[l:r]


def truncate_seq_first(tokens, max_seq_length):
    max_num_tokens = max_seq_length - 2
    trunc_tokens = list(tokens)
    if len(trunc_tokens) > max_num_tokens:
        trunc_tokens = trunc_tokens[:max_num_tokens]
    return trunc_tokens
