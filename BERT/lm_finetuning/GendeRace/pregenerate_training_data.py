from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm, trange
from tempfile import TemporaryDirectory
import shelve
from multiprocessing import Pool
from typing import List, Collection
from random import random, randrange, randint, shuffle, choice
from transformers.tokenization_bert import BertTokenizer
import pandas as pd
import numpy as np
import json
import collections
from constants import BERT_PRETRAINED_MODEL, POMS_GENDER_PRETRAIN_DATA_DIR, POMS_RACE_PRETRAIN_DATA_DIR, MAX_POMS_SEQ_LENGTH, POMS_RAW_DATA_DIR
from Timer import timer

WORDPIECE_PREFIX = "##"
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
MASK_TOKEN = "[MASK]"


EPOCHS = 5
MLM_PROB = 0.15
MAX_PRED_PER_SEQ = 30


class DocumentDatabase:
    def __init__(self, reduce_memory=False):
        if reduce_memory:
            self.temp_dir = TemporaryDirectory()
            self.working_dir = Path(self.temp_dir.name)
            self.document_shelf_filepath = self.working_dir / 'shelf.db'
            self.document_shelf = shelve.open(str(self.document_shelf_filepath),
                                              flag='n', protocol=-1)
            self.documents = None
        else:
            self.documents = []
            self.document_ids = []
            self.documents_labels = []
            self.document_shelf = None
            self.document_shelf_filepath = None
            self.temp_dir = None
        self.doc_lengths = []
        self.doc_cumsum = None
        self.cumsum_max = None
        self.reduce_memory = reduce_memory

    def add_document(self, document, label, unique_id):
        if not document:
            return
        if self.reduce_memory:
            current_idx = len(self.doc_lengths)
            self.document_shelf[str(current_idx)] = document
        else:
            self.documents.append(document)
            self.document_ids.append(unique_id)
            self.documents_labels.append(label)
        self.doc_lengths.append(len(document))

    def _precalculate_doc_weights(self):
        self.doc_cumsum = np.cumsum(self.doc_lengths)
        self.cumsum_max = self.doc_cumsum[-1]

    def sample_doc(self, current_idx, sentence_weighted=True):
        # Uses the current iteration counter to ensure we don't sample the same doc twice
        if sentence_weighted:
            # With sentence weighting, we sample docs proportionally to their sentence length
            if self.doc_cumsum is None or len(self.doc_cumsum) != len(self.doc_lengths):
                self._precalculate_doc_weights()
            rand_start = self.doc_cumsum[current_idx]
            rand_end = rand_start + self.cumsum_max - self.doc_lengths[current_idx]
            sentence_index = randrange(rand_start, rand_end) % self.cumsum_max
            sampled_doc_index = np.searchsorted(self.doc_cumsum, sentence_index, side='right')
        else:
            # If we don't use sentence weighting, then every doc has an equal chance to be chosen
            sampled_doc_index = (current_idx + randrange(1, len(self.doc_lengths))) % len(self.doc_lengths)
        assert sampled_doc_index != current_idx
        if self.reduce_memory:
            return self.document_shelf[str(sampled_doc_index)]
        else:
            return self.documents[sampled_doc_index]

    def __len__(self):
        return len(self.doc_lengths)

    def __getitem__(self, item):
        if self.reduce_memory:
            return self.document_shelf[str(item)]
        else:
            return self.documents[item], self.documents_labels[item], self.document_ids[item]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        if self.document_shelf is not None:
            self.document_shelf.close()
        if self.temp_dir is not None:
            self.temp_dir.cleanup()


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
    """Truncates a pair of sequences to a maximum sequence length. Lifted from Google's BERT repo."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


def truncate_seq(tokens, max_num_tokens):
    """Truncates a pair of sequences to a maximum sequence length. Lifted from Google's BERT repo."""
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



MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])


def generate_cand_indices(num_tokens: int, tokens: Collection, whole_word_mask) -> List[List[int]]:
    idx_list = np.random.permutation(list(set(range(1, num_tokens + 1, 1))))
    cand_indices = []
    for i in idx_list:
        if (whole_word_mask and len(cand_indices) >= 1 and tokens[i].startswith(WORDPIECE_PREFIX)):
            cand_indices[-1].append(i)
        else:
            cand_indices.append([i])
    return cand_indices


def create_masked_lm_predictions(tokens, cand_indices, num_to_mask, vocab_list):
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
    with several refactors to clean it up and remove a lot of unnecessary variables."""
    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indices:
        if len(masked_lms) >= num_to_mask:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_mask:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)
            ## if index is adjective
            masked_token = None
            # 80% of the time, replace with [MASK]
            if random() < 0.8:
                masked_token = MASK_TOKEN
            else:
                # 10% of the time, keep original
                if random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = choice(vocab_list)
            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
            tokens[index] = masked_token
            ## if index not adjective

    assert len(masked_lms) <= num_to_mask
    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    mask_indices = [p.index for p in masked_lms]
    masked_token_labels = [p.label for p in masked_lms]

    return tokens, mask_indices, masked_token_labels


def get_num_to_mask(max_predictions_per_seq: int, num_tokens: int, masked_lm_prob: float) -> int:
    return min(max_predictions_per_seq, max(1, int(round(num_tokens * masked_lm_prob))))


def create_instances_from_document(
        doc_database, doc_idx, max_seq_length, short_seq_prob,
        masked_lm_prob, max_predictions_per_seq, whole_word_mask, vocab_list):
    """This code is mostly a duplicate of the equivalent function from Google BERT's repo.
    However, we make some changes and improvements. Sampling is improved and no longer requires a loop in this function.
    Also, documents are sampled proportionally to the number of sentences they contain, which means each sentence
    (rather than each document) has an equal chance of being sampled as a false example for the NextSentence task."""
    document, label, unique_id = doc_database[doc_idx]
    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 2

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if random() < short_seq_prob:
        target_seq_length = randint(2, max_num_tokens)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.

    tokens = truncate_seq(document, max_num_tokens)

    assert len(tokens) >= 1

    tokens = tuple([CLS_TOKEN] + tokens + [SEP_TOKEN])
    num_tokens = len(tokens) - 2

    num_to_mask = get_num_to_mask(max_predictions_per_seq, num_tokens, masked_lm_prob)

    cand_indices = generate_cand_indices(num_tokens, tokens, whole_word_mask)

    instances = []
    num_masked = 0
    while num_masked < len(cand_indices) and num_masked < num_to_mask:
        instance_tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(list(tokens),
                                                                                              cand_indices[num_masked:],
                                                                                              num_to_mask, vocab_list)

        instance = {
            "unique_id": str(unique_id),
            "tokens": [str(i) for i in instance_tokens],
            "masked_lm_positions": [str(i) for i in masked_lm_positions],
            "masked_lm_labels": [str(i) for i in masked_lm_labels],
            "genderace_label": str(label)
        }

        instances.append(instance)

        num_masked += len(masked_lm_labels)

    return instances


def create_training_file(docs, vocab_list, args, epoch_num, output_dir):
    epoch_filename = output_dir / f"{BERT_PRETRAINED_MODEL}_epoch_{epoch_num}.json"
    num_instances = 0
    with epoch_filename.open('w') as epoch_file:
        for doc_idx in trange(len(docs), desc="Document"):
            doc_instances = create_instances_from_document(
                docs, doc_idx, max_seq_length=args.max_seq_len, short_seq_prob=args.short_seq_prob,
                masked_lm_prob=args.masked_lm_prob, max_predictions_per_seq=args.max_predictions_per_seq,
                whole_word_mask=args.do_whole_word_mask, vocab_list=vocab_list)
            doc_instances = [json.dumps(instance) for instance in doc_instances]
            for instance in doc_instances:
                epoch_file.write(instance + '\n')
                num_instances += 1
    metrics_file = output_dir / f"{BERT_PRETRAINED_MODEL}_epoch_{epoch_num}_metrics.json"
    with metrics_file.open('w') as metrics_file:
        metrics = {
            "num_training_examples": num_instances,
            "max_seq_len": args.max_seq_len
        }
        metrics_file.write(json.dumps(metrics))


@timer
def main():
    parser = ArgumentParser()
    parser.add_argument('--train_corpus', type=Path, required=False)
    parser.add_argument("--output_dir", type=Path, required=False)
    parser.add_argument("--bert_model", type=str, required=False, default=BERT_PRETRAINED_MODEL,
                        choices=["bert-base-uncased", "bert-large-uncased", "bert-base-cased",
                                 "bert-base-multilingual-uncased", "bert-base-chinese", "bert-base-multilingual-cased"])
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--do_whole_word_mask", action="store_true",
                        help="Whether to use whole word masking rather than per-WordPiece masking.")
    parser.add_argument("--reduce_memory", action="store_true",
                        help="Reduce memory usage for large datasets by keeping data on disc rather than in memory")

    parser.add_argument("--num_workers", type=int, default=EPOCHS,
                        help="The number of workers to use to write the files")
    parser.add_argument("--epochs_to_generate", type=int, default=EPOCHS,
                        help="Number of epochs of data to pregenerate")
    parser.add_argument("--max_seq_len", type=int, default=MAX_POMS_SEQ_LENGTH)
    parser.add_argument("--short_seq_prob", type=float, default=0.1,
                        help="Probability of making a short sentence as a training example")
    parser.add_argument("--masked_lm_prob", type=float, default=MLM_PROB,
                        help="Probability of masking each token for the LM task")
    parser.add_argument("--max_predictions_per_seq", type=int, default=MAX_PRED_PER_SEQ,
                        help="Maximum number of tokens to mask in each sequence")
    parser.add_argument("--treatment", type=str, required=True, default="gender",
                        help="Treatment can be: gender or race")
    parser.add_argument("--corpus_type", type=str, required=False, default="",
                        help="Corpus type can be: '', enriched, enriched_noisy, enriched_full")
    args = parser.parse_args()

    if args.num_workers > 1 and args.reduce_memory:
        raise ValueError("Cannot use multiple workers while reducing memory")
    args.epochs_to_generate = EPOCHS
    tokenizer = BertTokenizer.from_pretrained(BERT_PRETRAINED_MODEL, do_lower_case=bool(BERT_PRETRAINED_MODEL.endswith("uncased")))
    vocab_list = list(tokenizer.vocab.keys())

    if args.treatment == "gender":
        PRETRAIN_DATA_OUTPUT_DIR = Path(POMS_GENDER_PRETRAIN_DATA_DIR)
        treatment_column = "Gender"
        treatment_condition = "female"
    else:
        PRETRAIN_DATA_OUTPUT_DIR = Path(POMS_RACE_PRETRAIN_DATA_DIR)
        treatment_column = "Race"
        treatment_condition = "African-American"

    if "enriched" in args.corpus_type:
        DATASET_FILE = f"{POMS_RAW_DATA_DIR}/Equity-Evaluation-Corpus_{args.corpus_type}.csv"
        PRETRAIN_DATA_OUTPUT_DIR = PRETRAIN_DATA_OUTPUT_DIR / args.corpus_type
    else:
        DATASET_FILE = f"{POMS_RAW_DATA_DIR}/Equity-Evaluation-Corpus.csv"

    with DocumentDatabase(reduce_memory=args.reduce_memory) as docs:
        if "enriched" in args.corpus_type:
            df = pd.read_csv(DATASET_FILE, header=0, encoding='utf-8').set_index(keys="ID", drop=False).sort_index()
        else:
            df = pd.read_csv(DATASET_FILE, header=0, encoding='utf-8', converters={"ID": lambda i: int(i.split("-")[-1])}).set_index(keys="ID", drop=False).sort_index()
        df = df[df[treatment_column].notnull()]
        unique_ids = df["ID"]
        documents = df["Sentence"].apply(tokenizer.tokenize)
        genderace_labels = df[treatment_column].apply(lambda t: int(str(t) == treatment_condition))
        for doc, label, unique_id in tqdm(zip(documents, genderace_labels, unique_ids)):
            if doc:
                docs.add_document(doc, label, unique_id)  # If the last doc didn't end on a newline, make sure it still gets added
        if len(docs) <= 1:
            exit("ERROR: No document breaks were found in the input file! These are necessary to allow the script to "
                 "ensure that random NextSentences are not sampled from the same document. Please add blank lines to "
                 "indicate breaks between documents in your input file. If your dataset does not contain multiple "
                 "documents, blank lines can be inserted at any natural boundary, such as the ends of chapters, "
                 "sections or paragraphs.")

        output_dir = Path(PRETRAIN_DATA_OUTPUT_DIR)
        output_dir.mkdir(exist_ok=True, parents=True)

        if args.num_workers > 1:
            writer_workers = Pool(min(args.num_workers, args.epochs_to_generate))
            arguments = [(docs, vocab_list, args, idx, output_dir) for idx in range(args.epochs_to_generate)]
            writer_workers.starmap(create_training_file, arguments)
        else:
            for epoch in trange(args.epochs_to_generate, desc="Epoch"):
                create_training_file(docs, vocab_list, args, epoch, output_dir)


if __name__ == '__main__':
    main()
