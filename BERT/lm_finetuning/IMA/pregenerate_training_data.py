from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm, trange
from tempfile import TemporaryDirectory
import shelve
from constants import BERT_PRETRAINED_MODEL, SENTIMENT_RAW_DATA_DIR, SENTIMENT_IMA_DATA_DIR, MAX_SEQ_LENGTH, SENTIMENT_DOMAINS
from datasets.datasets_utils import TOKEN_SEPARATOR
from multiprocessing import Pool
from random import random, randrange, choice
from transformers.tokenization_bert import BertTokenizer
from itertools import zip_longest
from Timer import timer
from typing import List
import numpy as np
import json
import collections
import re

WORDPIECE_PREFIX = "##"
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
MASK_TOKEN = "[MASK]"
ADJ_POS_TAGS = ("ADJ", "ADV")

EPOCHS = 5
MLM_PROB = 0.15
MAX_PRED_PER_SEQ = 30


class POSTaggedDocumentDatabase:
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
            self.documents_pos_idx = []
            self.document_shelf = None
            self.document_shelf_filepath = None
            self.temp_dir = None
        self.doc_lengths = []
        self.doc_pos_lengths = []
        self.doc_cumsum = None
        self.cumsum_max = None
        self.reduce_memory = reduce_memory

    def add_document(self, document, doc_pos_idx):
        if not document:
            return
        if self.reduce_memory:
            current_idx = len(self.doc_lengths)
            self.document_shelf[str(current_idx)] = document
        else:
            self.documents.append(document)
            self.documents_pos_idx.append(doc_pos_idx)
        self.doc_lengths.append(len(document))
        self.doc_pos_lengths.append(len(doc_pos_idx))

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
            return self.documents[item], self.documents_pos_idx[item]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        if self.document_shelf is not None:
            self.document_shelf.close()
        if self.temp_dir is not None:
            self.temp_dir.cleanup()


def truncate_seq(tokens, max_num_tokens, doc_pos_idx):
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
    if l > 0 or r < len(tokens):
        trunc_doc_pos_idx = [i - l for i in doc_pos_idx if l <= i <= r]
    else:
        trunc_doc_pos_idx = list(doc_pos_idx)
    return trunc_tokens[l:r], trunc_doc_pos_idx


MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])


def create_masked_adj_predictions(tokens, tokens_pos_idx, cand_indices, num_to_mask, vocab_list):
    # Positive examples: M is on Adj x num_adj + word is Adj x num_adj
    # Negative examples: M is not on Adj x num_adj + word is not Adj x num_adj
    masked_lms = []
    covered_indexes = set()
    for i, index_set in enumerate(cand_indices):
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

    assert len(masked_lms) <= num_to_mask
    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    mask_indices = [p.index for p in masked_lms]
    masked_token_labels = [p.label for p in masked_lms]
    masked_token_adj_labels = [int(p.index in tokens_pos_idx) for p in masked_lms]

    return tokens, mask_indices, masked_token_labels, masked_token_adj_labels


def mlm_prob(num_adj: int, num_tokens: int, masked_lm_prob: float) -> int:
    return min(num_adj * 2, max(1, int(round(num_tokens * masked_lm_prob))))


# TODO: Experiment with num_adj * 2 (or num_adj*1.X if num_adj*2>=num_tokens)
def double_num_adj(num_adj: int, num_tokens: int, masked_adj_ratio: float) -> int:
    adj_ratio = float(num_adj) / num_tokens
    if adj_ratio <= masked_adj_ratio:
        return num_adj * 2
    else:
        return int(round(num_adj * (1 + (1 - adj_ratio))))


def generate_cand_indices(num_tokens: int, tokens_pos_idx: List[int]) -> List[List[int]]:
    adj_idx_list = np.random.permutation(tokens_pos_idx)
    non_adj_idx_list = np.random.permutation(list(set(range(1, num_tokens + 1, 1)) - set(tokens_pos_idx)))
    cand_indices = []
    for i, j in zip_longest(adj_idx_list, non_adj_idx_list, fillvalue=None):
        if i:
            cand_indices.append([i])
        if j:
            cand_indices.append([j])
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word. When a word has been split into
        # WordPieces, the first token does not have any marker and any subsequence
        # tokens are prefixed with ##. So whenever we see the ## token, we
        # append it to the previous set of word indexes.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        # if (whole_word_mask and len(cand_indices) >= 1 and token.startswith("##")):
        #     cand_indices[-1].append(i)
        # else:
        #     cand_indices.append([i])
    return cand_indices


def create_instances_from_document(
        doc_database, doc_idx, max_seq_length, short_seq_prob,
        masked_lm_prob, max_predictions_per_seq, whole_word_mask, vocab_list):
    """This code is mostly a duplicate of the equivalent function from Google BERT's repo.
    However, we make some changes and improvements. Sampling is improved and no longer requires a loop in this function.
    Also, documents are sampled proportionally to the number of sentences they contain, which means each sentence
    (rather than each document) has an equal chance of being sampled as a false example for the NextSentence task."""
    document, doc_pos_idx = doc_database[doc_idx]
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
        target_seq_length = max_num_tokens / 2

    tokens_a, tokens_pos_idx_list = truncate_seq(document, max_num_tokens, doc_pos_idx)

    assert len(tokens_a) >= 1

    tokens = tuple([CLS_TOKEN] + tokens_a + [SEP_TOKEN])
    # The segment IDs are 0 for the [CLS] token, the A tokens and the first [SEP]
    # They are 1 for the B tokens and the final [SEP]
    # segment_ids = [0 for _ in range(len(tokens_a) + 2)]
    tokens_pos_idx = [i + 1 for i in tokens_pos_idx_list]
    num_adj = len(tokens_pos_idx)
    num_tokens = len(tokens) - 2
    num_non_adj = num_tokens - num_adj

    # Currently we follow original MLM training regime where at most 15% of tokens in sequence are masked.
    # For each adjective we add a non-adjective to be masked, to preserve balanced classes
    # This means that if in a given sequence there are more than 15% adjectives,
    # we produce sequences where not all adjectives are masked and they will appear in context
    # We produce as many such sequences as needed in order to mask all adjectives in original sequence during training
    # if num_to_mask > max_predictions_per_seq:
    #     print(f"{num_to_mask} is more than max per seq of {max_predictions_per_seq}")
    # if num_to_mask > int(round(len(tokens) * masked_lm_prob)):
    #     print(f"{num_to_mask} is more than {masked_lm_prob} of {num_tokens}")
    # if num_to_mask > len(tokens):
    #     print(f"{num_to_mask} is more than {num_tokens}")

    num_to_mask = mlm_prob(num_adj, num_tokens, masked_lm_prob)
    # num_to_mask = double_num_adj(num_adj, num_tokens, 0.4)

    cand_indices = generate_cand_indices(num_tokens, tokens_pos_idx)

    instances = []
    num_adj_masked = 0
    num_masked = 0
    while num_masked < len(cand_indices) and num_adj_masked < num_adj:
        instance_tokens, masked_lm_positions, masked_lm_labels, masked_adj_labels = create_masked_adj_predictions(
            list(tokens), tokens_pos_idx, cand_indices[num_masked:], num_to_mask, vocab_list)

        instance = {
            "tokens": [str(i) for i in instance_tokens],
            "masked_lm_positions": [str(i) for i in masked_lm_positions],
            "masked_lm_labels": [str(i) for i in masked_lm_labels],
            "masked_adj_labels": [str(i) for i in masked_adj_labels]
        }
        instances.append(instance)

        num_adj_masked += sum(masked_adj_labels)
        num_masked = len(masked_lm_labels)

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
    print("\nTotal Number of training instances:", num_instances)


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
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--short_seq_prob", type=float, default=0.1,
                        help="Probability of making a short sentence as a training example")
    parser.add_argument("--masked_lm_prob", type=float, default=MLM_PROB,
                        help="Probability of masking each token for the LM task")
    parser.add_argument("--max_predictions_per_seq", type=int, default=MAX_PRED_PER_SEQ,
                        help="Maximum number of tokens to mask in each sequence")

    args = parser.parse_args()

    if args.num_workers > 1 and args.reduce_memory:
        raise ValueError("Cannot use multiple workers while reducing memory")
    args.epochs_to_generate = EPOCHS
    tokenizer = BertTokenizer.from_pretrained(BERT_PRETRAINED_MODEL, do_lower_case=bool(BERT_PRETRAINED_MODEL.endswith("uncased")))
    vocab_list = list(tokenizer.vocab.keys())
    for domain in SENTIMENT_DOMAINS:
        print(f"\nGenerating data for domain: {domain}")
        DATASET_FILE = f"{SENTIMENT_RAW_DATA_DIR}/{domain}/{domain}UN_tagged.txt"
        DATA_OUTPUT_DIR = Path(SENTIMENT_IMA_DATA_DIR) / domain
        with POSTaggedDocumentDatabase(reduce_memory=args.reduce_memory) as docs:
            with open(DATASET_FILE, "r") as dataset:
                for line in tqdm(dataset):
                    tagged_tokens = []
                    adj_adv_idx = []
                    adj_adv_tokens = []
                    line_tokens = []
                    for i, token_pos in enumerate(line.strip().split(TOKEN_SEPARATOR)):
                        token_pos_match = re.match("(.*)_([A-Z]+)", token_pos)
                        if token_pos_match:
                            token, pos = token_pos_match.group(1), token_pos_match.group(2)
                            tagged_tokens.append((token, pos))
                            if pos in ADJ_POS_TAGS:
                                adj_adv_tokens.append((i, token))
                            line_tokens.append(token)
                    # line_words = re.sub("_[A-Z]+", "", line)
                    doc = tokenizer.tokenize(TOKEN_SEPARATOR.join(line_tokens))
                    if doc:
                        if len(doc) == len(tagged_tokens):
                            adj_adv_idx = [i for i, _ in adj_adv_tokens]
                        else:
                            adj_token_idx = 0
                            for j, bert_token in enumerate(doc):
                                if adj_token_idx == len(adj_adv_tokens):
                                    break
                                adj_token = adj_adv_tokens[adj_token_idx][1]
                                if bert_token == adj_token:
                                    adj_adv_idx.append(j)
                                    adj_token_idx += 1
                                elif bert_token in adj_token:
                                    adj_adv_idx.append(j)
                                    # if args.do_whole_word_mask:
                                    k = 1
                                    while j + k < len(doc) and doc[j + k].startswith(WORDPIECE_PREFIX):
                                        adj_adv_idx.append(j + k)
                                        k += 1
                                    adj_token_idx += 1
                        docs.add_document(tuple(doc), tuple(adj_adv_idx))  # If the last doc didn't end on a newline, make sure it still gets added
                if len(docs) <= 1:
                    exit("ERROR: No document breaks were found in the input file! These are necessary to allow the script to "
                         "ensure that random NextSentences are not sampled from the same document. Please add blank lines to "
                         "indicate breaks between documents in your input file. If your dataset does not contain multiple "
                         "documents, blank lines can be inserted at any natural boundary, such as the ends of chapters, "
                         "sections or paragraphs.")

            output_dir = DATA_OUTPUT_DIR
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
