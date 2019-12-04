# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Modified by Nadav Oved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Extract pre-computed feature vectors from a PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import logging
import re
from pathlib import Path
from random import random

import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers.tokenization_bert import BertTokenizer
from transformers.modeling_bert import BertModel
from BERT.lm_finetuning.pregenerate_training_data import truncate_seq, CLS_TOKEN, SEP_TOKEN, TOKEN_SEPARATOR
from constants import BERT_PRETRAINED_MODEL, RANDOM_SEED, MAX_SEQ_LENGTH, SENTIMENT_MODE_DATA_DIR,\
    OOB_PRETRAINED_MODEL, SENTIMENT_RAW_DATA_DIR, DOMAIN, MODE, FINAL_PRETRAINED_MODEL
from utils import init_logger
from Timer import timer
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch.nn as nn
import json


### Constants
BATCH_SIZE = 8  # Number of interviews in batch
FP16 = False
PRETRAINED_MODEL = FINAL_PRETRAINED_MODEL
DATASET_DIR = f"{SENTIMENT_RAW_DATA_DIR}/{DOMAIN}"
TRAIN_SET = f"{DATASET_DIR}/train.csv"
DEV_SET = f"{DATASET_DIR}/dev.csv"
TEST_SET = f"{DATASET_DIR}/test.csv"
OUTPUT_DIR = f"{SENTIMENT_MODE_DATA_DIR}/{DOMAIN}/features"

# logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
#                     datefmt = '%m/%d/%Y %H:%M:%S',
#                     level = logging.INFO)
# logger = logging.getLogger(__name__)

logger = init_logger("extract_features", f"{SENTIMENT_RAW_DATA_DIR}/extract_features.log")


class InputExample:

    def __init__(self, unique_id, text, text_no_adj, label):
        self.unique_id = unique_id
        self.text = text
        self.text_no_adj = text_no_adj
        self.label = label


class InputFeatures:
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask


def truncate_seq(tokens, max_seq_length):
    """Truncates a sequence to a maximum sequence length. Lifted from Google's BERT repo."""
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


def tokenize_and_build_features(review_unique_id, review_text, tokenizer, max_seq_length):
    review_tokens = tokenizer.tokenize(review_text)

    tokens = tuple([CLS_TOKEN] + truncate_seq(review_tokens, max_seq_length) + [SEP_TOKEN])

    review_len = len(tokens) - 2

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length

    return InputFeatures(unique_id=review_unique_id, tokens=tokens, input_ids=input_ids, input_mask=input_mask), review_len


@timer(logger=logger)
def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputFeature`s."""
    ### Modify code below to read from your dataset format
    reviews_features_list = list()
    reviews_seq_lengths = list()
    reviews_no_adj_features_list = list()
    reviews_no_adj_seq_lengths = list()
    for review_idx, review_example in enumerate(examples):
        review_features, review_len = tokenize_and_build_features(review_example.unique_id, review_example.text, tokenizer, seq_length)
        review_no_adj_features, review_no_adj_len = tokenize_and_build_features(review_example.unique_id, review_example.text_no_adj, tokenizer, seq_length)
        if review_idx < 10:
            logger.info(f"*** Example {review_idx + 1}***")
            logger.info("unique_id: %s" % review_features.unique_id)
            logger.info("tokens: %s" % " ".join([str(x) for x in review_features.tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in review_features.input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in review_features.input_mask]))
        reviews_features_list.append(review_features)
        reviews_seq_lengths.append(review_len)
        reviews_no_adj_features_list.append(review_no_adj_features)
        reviews_no_adj_seq_lengths.append(review_no_adj_len)
    print_seq_lengths_stats(reviews_seq_lengths, seq_length)
    print_seq_lengths_stats(reviews_no_adj_seq_lengths, seq_length)
    return reviews_features_list, reviews_no_adj_features_list


def print_seq_lengths_stats(text_seq_lengths, max_seq_length):
    logger.info(f"Num Sequences: {len(text_seq_lengths)}")
    logger.info(f"Minimum Sequence Length: {np.min(text_seq_lengths)}")
    logger.info(f"Average Sequence Length: {np.mean(text_seq_lengths)}")
    logger.info(f"Median Sequence Length: {np.median(text_seq_lengths)}")
    logger.info(f"99th Percentile Sequence Length: {np.percentile(text_seq_lengths, 99)}")
    logger.info(f"Maximum Sequence Length: {np.max(text_seq_lengths)}")
    logger.info(f"Num of over Maximum Sequence Length: {len([i for i in text_seq_lengths if i >= max_seq_length])}")


@timer(logger=logger)
def read_examples(input_file):
    """Read a list of `InputExample`s from an input file."""
    ### Modify code below to read from your dataset format
    df = pd.read_csv(input_file, header=0, encoding='utf-8')
    return df.apply(lambda row: InputExample(unique_id=row.iloc[0], text=row["review"], text_no_adj=row["no_adj_review"], label=row["label"]), axis=1).tolist()


@timer(logger=logger)
def extract_features(dataset, args):
    dataset_file = f"{args.input_dir}/{dataset}.csv"
    output_path = Path(f"{args.output_dir}/{dataset}")
    output_path.mkdir(parents=True, exist_ok=True)
    errors = list()

    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        # device = get_free_gpu()
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {} distributed training: {}".format(device, n_gpu, bool(args.local_rank != -1)))

    # layer_indexes = [int(x) for x in args.layers.split(",")]

    tokenizer = BertTokenizer.from_pretrained(BERT_PRETRAINED_MODEL,
                                              do_lower_case=bool(BERT_PRETRAINED_MODEL.endswith("uncased")))

    reviews_examples = read_examples(dataset_file)

    reviews_features_lists = convert_examples_to_features(examples=reviews_examples, seq_length=MAX_SEQ_LENGTH, tokenizer=tokenizer)

    if PRETRAINED_MODEL != OOB_PRETRAINED_MODEL:
        fine_tuned_state_dict = torch.load(PRETRAINED_MODEL)
        model = BertModel.from_pretrained(BERT_PRETRAINED_MODEL, state_dict=fine_tuned_state_dict)
    else:
        model = BertModel.from_pretrained(BERT_PRETRAINED_MODEL)
    model.to(device)

    if args.local_rank != -1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    # elif n_gpu > 1:
    #     model = nn.DataParallel(model)

    for features_list, features_type in zip(reviews_features_lists, ("", "_no_adj")):
        input_ids_list = list()
        input_masks_list = list()
        for f in features_list:
            input_ids_list.append(f.input_ids)
            input_masks_list.append(f.input_mask)
        all_input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        all_input_mask = torch.tensor(input_masks_list, dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_example_index)
        if args.local_rank == -1:
            eval_sampler = SequentialSampler(eval_data)
        else:
            eval_sampler = DistributedSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=BATCH_SIZE)

        model.eval()

        reviews_output_dict = collections.OrderedDict()
        with torch.no_grad():
            for input_ids, input_mask, example_indices in tqdm(eval_dataloader):
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)

                last_encoder_layer, pooled_output = model(input_ids, attention_mask=input_mask)

                last_layer_output = last_encoder_layer.detach().cpu()
                # pooled_output_detached = pooled_output.detach().cpu()

                ### Modify code below to your needs of organizing and saving BERT model outputs
                for batch_idx, example_index in enumerate(example_indices):
                    feature = features_list[example_index.item()]
                    unique_id = feature.unique_id
                    example_output_dict = collections.OrderedDict()
                    example_output_dict["unique_id"] = str(unique_id)
                    last_layer_output_example = last_layer_output[batch_idx]
                    example_output_dict["tokens"] = TOKEN_SEPARATOR.join([str(t) for t in feature.tokens])
                    if len(example_output_dict.keys()) <= 0:
                        errors.append(f"{unique_id}")
                    reviews_output_dict[unique_id] = example_output_dict
                    output_file = f"{unique_id}_{BERT_PRETRAINED_MODEL}-review_encodings{features_type}"
                    # logger.info(f"Saving {output_file} to {output_path}")
                    torch.save(last_layer_output_example, output_path / f"{output_file}.pt")
                    with open(output_path / f"{output_file}.json", "w") as jsonfile:
                        json.dump(example_output_dict, jsonfile)
        # send_email([f"Saved BERT Encodings to: {output_file}", "\nErrors:"] + errors, "BERT Encodings Extraction")


@timer(logger=logger)
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input_dir", default=DATASET_DIR, type=str, required=False)
    parser.add_argument("--output_dir", default=OUTPUT_DIR, type=str, required=False)
    parser.add_argument("--bert_model", default=BERT_PRETRAINED_MODEL, type=str, required=False,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    ## Other parameters
    parser.add_argument("--layers", default="-1,-2,-3,-4", type=str)
    parser.add_argument("--max_seq_length", default=MAX_SEQ_LENGTH, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                            "than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--batch_size", default=BATCH_SIZE, type=int, help="Batch size for predictions.")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on GPUs")
    parser.add_argument("--no_cuda", action='store_true', help="Whether not to use CUDA when available")

    args = parser.parse_args()

    for dataset in ("train", "dev", "test"):
        extract_features(dataset, args)


if __name__ == "__main__":
    main()
