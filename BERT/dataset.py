from random import random
from torch.utils.data import TensorDataset, Dataset
from tqdm import tqdm
from transformers.tokenization_bert import BertTokenizer
from BERT.lm_finetuning.MLM.pregenerate_training_data import CLS_TOKEN, SEP_TOKEN
from constants import BERT_PRETRAINED_MODEL, MAX_SENTIMENT_SEQ_LENGTH
import pandas as pd
import numpy as np
import torch

### Constants
PAD_ID = 0


class BertTextClassificationDataset(Dataset):
    def __init__(self, data_path: str, treatment: str, subset: str, text_column: str, label_column: str,
                 bert_pretrained_model: str = BERT_PRETRAINED_MODEL, max_seq_length: int = MAX_SENTIMENT_SEQ_LENGTH):
        super().__init__()
        if subset not in ("train", "dev", "test"):
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

    def read_examples(self):
        """Read a list of `InputExample`s from an input file."""
        df = pd.read_csv(self.dataset_file, header=0, encoding='utf-8')
        return df.apply(lambda row:
                        InputExample(unique_id=int(row.iloc[0]),
                                     text=str(row[self.text_column]),
                                     label=int(row[self.label_column])),
                        axis=1).tolist()

    def convert_examples_to_features(self, examples):
        """Loads a data file into a list of `InputFeature`s."""
        features_list = list()
        labels_list = list()
        # seq_lengths = list()
        for i, example in tqdm(enumerate(examples), total=len(examples), desc=f"{self.subset}-convert_examples_to_features"):
            features, example_len = self.tokenize_and_pad_sequence(example)
            # if i < 10:
            #     logger.info(f"*** Example {i + 1}***")
            #     logger.info("unique_id: %s" % features.unique_id)
            #     logger.info("tokens: %s" % " ".join([str(x) for x in features.tokens]))
            #     logger.info("input_ids: %s" % " ".join([str(x) for x in features.input_ids]))
            #     logger.info("input_mask: %s" % " ".join([str(x) for x in features.input_mask]))
            features_list.append(features)
            labels_list.append(InputLabel(unique_id=example.unique_id, label=example.label))
            # seq_lengths.append(example_len)
        # print_seq_lengths_stats(seq_lengths, self.max_seq_length)
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
            input_ids.append(PAD_ID)
            input_mask.append(PAD_ID)

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length

        return InputFeatures(unique_id=example.unique_id, tokens=tokens,
                             input_ids=input_ids, input_mask=input_mask), example_len

    @staticmethod
    def create_tensor_dataset(features, labels):
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
