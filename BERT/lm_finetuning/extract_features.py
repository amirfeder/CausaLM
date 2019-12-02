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

import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers.tokenization_bert import BertTokenizer
from transformers.modeling_bert import BertModel
from BERT.lm_finetuning.finetune_on_pregenerated import PregeneratedPOSTaggedDataset
from constants import BERT_PRETRAINED_MODEL, RANDOM_SEED, IMA_DATA_DIR, MAX_SEQ_LENGTH
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
FINE_TUNED_MODEL = f"{IMA_DATA_DIR}/books/model/pytorch_model.bin"


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):

    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


@timer(logger=logger)
def convert_examples_to_features(examples, seq_length, tokenizer, unique_id_gen):
    """Loads a data file into a list of `InputFeature`s."""
    ### Modify code below to read from your dataset format
    max_num_tokens = seq_length - 3
    overlap_size = 5
    interviews_features = list()
    interviews_seq_lengths = list()
    for interview_idx, interview_examples in enumerate(examples):
        qa_features = list()
        qa_instances = list()
        for qa_index, qa_example in enumerate(interview_examples):
            tokens_a = tokenizer.tokenize(qa_example.text_a)
            q_len = len(tokens_a)
            tokens_b = None
            if qa_example.text_b:
                tokens_b = tokenizer.tokenize(qa_example.text_b)

            if tokens_b:
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                a_len = len(tokens_b)
                total_length = q_len + a_len
                if q_len < 1:
                    tokens_a = tokens_b
                    tokens_b = []
                    q_len = a_len
                    a_len = 0
                if q_len >= max_num_tokens:
                    all_tokens = tokens_a + tokens_b
                    while total_length > max_num_tokens:
                        current_tokens = all_tokens[:max_num_tokens]
                        tokens_a = current_tokens[:max_num_tokens // 2]
                        tokens_b = current_tokens[max_num_tokens // 2:]
                        qa_instances.append((qa_index, tokens_a, tokens_b))
                        all_tokens = all_tokens[max_num_tokens-overlap_size:]
                        total_length = len(all_tokens)
                    if total_length >= 1:
                        qa_instances.append((qa_index, all_tokens, []))
                elif total_length > max_num_tokens:
                    slice_size = max_num_tokens - q_len
                    while total_length > max_num_tokens:
                        current_tokens = tokens_b[:slice_size]
                        qa_instances.append((qa_index, tokens_a, current_tokens))
                        tokens_b = tokens_b[slice_size-overlap_size:]
                        total_length = q_len + len(tokens_b)
                    if len(tokens_b) >= 1:
                        qa_instances.append((qa_index, tokens_a, tokens_b))
                else:
                    qa_instances.append((qa_index, tokens_a, tokens_b))
                # _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                # max_num_tokens = seq_length - 2
                while len(tokens_a) > max_num_tokens:
                    current_tokens = tokens_a[:max_num_tokens]
                    tokens_a = tokens_a[max_num_tokens-overlap_size:]
                    qa_instances.append((qa_index, current_tokens, []))
                if len(tokens_a) >= 1:
                    qa_instances.append((qa_index, tokens_a, []))

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids:   0   0  0    0    0     0      0   0    1  1  1   1  1   1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids:   0   0   0   0  0     0   0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambigiously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
        for idx, (qa_index, tokens_a, tokens_b) in enumerate(qa_instances):
            unique_id = unique_id_gen.get_unique_id((interview_idx, qa_index, idx-qa_index))
            tokens = []
            input_type_ids = []
            tokens.append("[CLS]")
            input_type_ids.append(0)
            for token in tokens_a:
                tokens.append(token)
                input_type_ids.append(0)
            tokens.append("[SEP]")
            input_type_ids.append(0)

            if tokens_b:
                for token in tokens_b:
                    tokens.append(token)
                    input_type_ids.append(1)
                tokens.append("[SEP]")
                input_type_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < seq_length:
                input_ids.append(0)
                input_mask.append(0)
                input_type_ids.append(0)

            assert len(input_ids) == seq_length
            assert len(input_mask) == seq_length
            assert len(input_type_ids) == seq_length

            if interview_idx < 10 and qa_index < 2:
                logger.info(f"*** Example {interview_idx + 1}:{qa_index + 1}:{idx + 1}***")
                logger.info("unique_id: %s" % unique_id)
                logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info("input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

            qa_features.append(
                InputFeatures(
                    unique_id=unique_id,
                    tokens=tokens,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    input_type_ids=input_type_ids))
            interviews_seq_lengths.append(len(tokens))
        interviews_features.append(qa_features)
    print_seq_lengths_stats(interviews_seq_lengths, seq_length)
    return interviews_features


def print_seq_lengths_stats(text_seq_lengths, max_seq_length):
    print(f"Num Sequences: {len(text_seq_lengths)}")
    print(f"Minimum Sequence Length: {np.min(text_seq_lengths)}")
    print(f"Average Sequence Length: {np.mean(text_seq_lengths)}")
    print(f"Median Sequence Length: {np.median(text_seq_lengths)}")
    print(f"99th Percentile Sequence Length: {np.percentile(text_seq_lengths, 99)}")
    print(f"Maximum Sequence Length: {np.max(text_seq_lengths)}")
    print(f"Num of over Maximum Sequence Length: {len([i for i in text_seq_lengths if i >= max_seq_length])}")


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


@timer(logger=logger)
def read_examples(input_file, unique_id_gen):
    """Read a list of `InputExample`s from an input file."""
    ### Modify code below to read from your dataset format
    df = pd.read_csv(input_file, header=0, usecols=["interviews"], encoding='utf-8')
    dataset = df.interviews.tolist()
    interview_examples = list()
    for interview_id, interview_str in enumerate(dataset):
        input_examples = list()
        interview_str = interview_str.strip()
        q_a_list = interview_str.split(Q_A_SEPARATOR)
        if not q_a_list[0]:
            q_a_list.pop(0)
        q_a_id = 0
        for q_a in q_a_list:
            q_a = re.sub(LINE_SEPARATOR, "", q_a)
            q_a = re.sub(Q_TOKEN, "", q_a)
            q_a = re.sub(SENTENCE_SEPARATOR, TOKEN_SEPARATOR, q_a)
            q_a_sep = q_a.split(A_TOKEN)
            q_str = q_a_sep.pop(0).strip()
            if not q_a_sep:
                unique_id = unique_id_gen.get_unique_id((interview_id, q_a_id))
                input_examples.append(InputExample(unique_id=unique_id, text_a=q_str, text_b=""))
                q_a_id += 1
            else:
                while q_a_sep:
                    a_str = q_a_sep.pop(0).strip()
                    unique_id = unique_id_gen.get_unique_id((interview_id, q_a_id))
                    input_examples.append(InputExample(unique_id=unique_id, text_a=q_str, text_b=a_str))
                    q_a_id += 1
        interview_examples.append(input_examples)
    return interview_examples


@timer(logger=logger)
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input_file", default=ALL_PBP_GAME_TEXT_METRICS, type=str, required=False)
    parser.add_argument("--output_file", default=None, type=str, required=False)
    parser.add_argument("--bert_model", default=None, type=str, required=False,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    ## Other parameters
    parser.add_argument("--layers", default="-1,-2,-3,-4", type=str)
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                            "than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for predictions.")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on GPUs")
    parser.add_argument("--no_cuda", action='store_true', help="Whether not to use CUDA when available")

    args = parser.parse_args()

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

    tokenizer = BertTokenizer.from_pretrained(BERT_PRETRAINED_MODEL, do_lower_case=bool(BERT_PRETRAINED_MODEL.endswith("uncased")))

    unique_id_mappings = UniqueIDGenerator(QA_UNIQUE_ID_MAPPING)

    interviews_examples = read_examples(ALL_PBP_GAME_TEXT_METRICS, unique_id_mappings)

    interviews_features = convert_examples_to_features(examples=interviews_examples, seq_length=MAX_SEQ_LENGTH,
                                                       tokenizer=tokenizer, unique_id_gen=unique_id_mappings)

    unique_id_mappings.save_mappings()
    # unique_id_to_feature = {}
    # for feature in features:
    #     unique_id_to_feature[feature.unique_id] = feature
    if FINE_TUNED_MODEL:
        fine_tuned_state_dict = torch.load(FINE_TUNED_MODEL)
        model = BertModel.from_pretrained(BERT_PRETRAINED_MODEL, state_dict=fine_tuned_state_dict)
    else:
        model = BertModel.from_pretrained(BERT_PRETRAINED_MODEL)
    model.to(device)

    if args.local_rank != -1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    elif n_gpu > 1:
        model = nn.DataParallel(model)

    for interview_idx, features in enumerate(tqdm(interviews_features)):
        input_ids_list = list()
        input_type_ids_list = list()
        input_masks_list = list()
        for f in features:
            input_ids_list.append(f.input_ids)
            input_type_ids_list.append(f.input_type_ids)
            input_masks_list.append(f.input_mask)
        all_input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        all_input_type_ids = torch.tensor(input_type_ids_list, dtype=torch.long)
        all_input_mask = torch.tensor(input_masks_list, dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_type_ids, all_input_mask, all_example_index)
        if args.local_rank == -1:
            eval_sampler = SequentialSampler(eval_data)
        else:
            eval_sampler = DistributedSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=MINI_BATCH_SIZE)

        model.eval()

        interview_output_dict = collections.OrderedDict()
        interview_tensors_list = list()

        for input_ids, input_type_ids, input_mask, example_indices in tqdm(eval_dataloader):
            input_ids = input_ids.to(device)
            input_type_ids = input_type_ids.to(device)
            input_mask = input_mask.to(device)

            last_encoder_layer, pooled_output = model(input_ids, token_type_ids=input_type_ids,
                                                      attention_mask=input_mask, output_all_encoded_layers=False)

            last_layer_output = last_encoder_layer.detach().cpu()
            pooled_output_detached = pooled_output.detach().cpu()

            ### Modify code below to your needs of organizing and saving BERT model outputs
            for batch_idx, example_index in enumerate(example_indices):
                feature = features[example_index.item()]
                unique_id = feature.unique_id
                interview_id, qa_id, segment_id = unique_id_mappings.recover_ids_from_unique_id(unique_id)
                assert interview_id == interview_idx, f"Interview ID mismatch! Expected {interview_idx} but got {interview_id}"
                example_output_dict = collections.OrderedDict()
                example_output_dict["qa_id"] = qa_id
                example_output_dict["segment_id"] = segment_id
                example_output_dict["unique_id"] = unique_id
                last_layer_output_example = last_layer_output[batch_idx]
                example_output_dict["tokens"] = TOKEN_SEPARATOR.join([str(t) for t in feature.tokens])
                if len(example_output_dict.keys()) <= 0:
                    errors.append(f"{interview_id}, {qa_id}, {segment_id}, {unique_id}")
                interview_output_dict[unique_id] = example_output_dict
                interview_tensors_list.append(last_layer_output_example[0])
        if len(interview_output_dict.keys()) < len(eval_sampler) or len(interview_tensors_list) < len(eval_sampler):
            errors.append(f"{interview_idx}")
        interview_output_tensor = torch.stack(interview_tensors_list)
        if FINE_TUNED_MODEL:
            output_file = f"{BERT_QA_ENCODINGS_DIR}{interview_idx}_{BERT_PRETRAINED_MODEL}-fine-tuned_QA_encodings_{DATASET_VERSION}"
        else:
            output_file = f"{BERT_QA_ENCODINGS_DIR}{interview_idx}_{BERT_PRETRAINED_MODEL}_QA_encodings_{DATASET_VERSION}"
        torch.save(interview_output_tensor, f"{output_file}.pt")
        with open(f"{output_file}.json", "w") as jsonfile:
            json.dump(interview_output_dict, jsonfile)
    send_email([f"Saved BERT Encodings to: {output_file}", "\nErrors:"] + errors, "BERT Encodings Extraction")


if __name__ == "__main__":
    main()