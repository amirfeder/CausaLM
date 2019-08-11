from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm, trange
from utils import UniqueIDGenerator
from tempfile import TemporaryDirectory
from constants import Q_A_SEPARATOR, Q_TOKEN, A_TOKEN, SENTENCE_SEPARATOR, TOKEN_SEPARATOR, LINE_SEPARATOR
from bert_constants import ALL_PBP_GAME_TEXT_METRICS, BERT_PRETRAINED_MODEL, MAX_QA_SEQ_LENGTH, BERT_QA_FINE_TUNE_DATA_DIR,\
    QA_UNIQUE_ID_MAPPING
from extract_features import print_seq_lengths_stats
from random import random, randrange, randint, shuffle, choice, sample
from pytorch_pretrained_bert.tokenization import BertTokenizer
import numpy as np
import pandas as pd
import shelve
import json
import re

EPOCHS = 1


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
            self.document_shelf = None
            self.document_shelf_filepath = None
            self.temp_dir = None
        self.doc_lengths = []
        self.doc_cumsum = None
        self.cumsum_max = None
        self.reduce_memory = reduce_memory

    def add_document(self, document):
        if not document:
            return
        if self.reduce_memory:
            current_idx = len(self.doc_lengths)
            self.document_shelf[str(current_idx)] = document
        else:
            self.documents.append(document)
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
            return self.documents[item]

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


def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, vocab_list):
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
    with several refactors to clean it up and remove a lot of unnecessary variables."""
    cand_indices = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        cand_indices.append(i)

    num_to_mask = min(max_predictions_per_seq,
                      max(1, int(round(len(tokens) * masked_lm_prob))))
    shuffle(cand_indices)
    mask_indices = sorted(sample(cand_indices, num_to_mask))
    masked_token_labels = []
    for index in mask_indices:
        # 80% of the time, replace with [MASK]
        if random() < 0.8:
            masked_token = "[MASK]"
        else:
            # 10% of the time, keep original
            if random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                masked_token = choice(vocab_list)
        masked_token_labels.append(tokens[index])
        # Once we've saved the true label for that token, we can overwrite it with the masked version
        tokens[index] = masked_token

    return tokens, mask_indices, masked_token_labels


def create_instances_from_document(
        doc_database, doc_idx, max_seq_length, short_seq_prob,
        masked_lm_prob, max_predictions_per_seq, vocab_list):
    """This code is mostly a duplicate of the equivalent function from Google BERT's repo.
    However, we make some changes and improvements. Sampling is improved and no longer requires a loop in this function.
    Also, documents are sampled proportionally to the number of sentences they contain, which means each sentence
    (rather than each document) has an equal chance of being sampled as a false example for the NextSentence task."""
    document = doc_database[doc_idx]
    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3

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
    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = randrange(1, len(current_chunk))

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = []

                # Random next
                if len(current_chunk) == 1 or random() < 0.5:
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)

                    # Sample a random document, with longer docs being sampled more frequently
                    random_document = doc_database.sample_doc(current_idx=doc_idx, sentence_weighted=True)

                    random_start = randrange(0, len(random_document))
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j])
                        if len(tokens_b) >= target_b_length:
                            break
                    # We didn't actually use these segments so we "put them back" so
                    # they don't go to waste.
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                # Actual next
                else:
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
                # The segment IDs are 0 for the [CLS] token, the A tokens and the first [SEP]
                # They are 1 for the B tokens and the final [SEP]
                segment_ids = [0 for _ in range(len(tokens_a) + 2)] + [1 for _ in range(len(tokens_b) + 1)]

                tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(
                    tokens, masked_lm_prob, max_predictions_per_seq, vocab_list)

                instance = {
                    "tokens": tokens,
                    "segment_ids": segment_ids,
                    "is_random_next": is_random_next,
                    "masked_lm_positions": masked_lm_positions,
                    "masked_lm_labels": masked_lm_labels}
                instances.append(instance)
            current_chunk = []
            current_length = 0
        i += 1

    return instances


def parse_interviews_to_docs(docs, dataset, tokenizer, unique_id_gen):
    seq_lengths = list()
    for interview_id, interview_str in enumerate(tqdm(dataset)):
        doc = list()
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
            q_tokens = tokenizer.tokenize(q_str)
            if not q_a_sep:
                unique_id = unique_id_gen.get_unique_id((interview_id, q_a_id))
                doc.append((unique_id, q_tokens, [], len(q_tokens)))
                q_a_id += 1
            else:
                while q_a_sep:
                    a_str = q_a_sep.pop(0).strip()
                    a_tokens = tokenizer.tokenize(a_str)
                    total_length = len(q_tokens) + len(a_tokens)
                    seq_lengths.append(total_length)
                    unique_id = unique_id_gen.get_unique_id((interview_id, q_a_id))
                    doc.append((unique_id, q_tokens, a_tokens, total_length))
                    q_a_id += 1
        # if len(doc) == 0:
        #    print(f"{interview_id}: {q_a_list}")
        docs.add_document(doc)
    # assert len(docs) == len(dataset)
    print_seq_lengths_stats(seq_lengths, MAX_QA_SEQ_LENGTH)
    return min(max(seq_lengths), MAX_QA_SEQ_LENGTH)


def create_instance_from_tokens(tokens_a, tokens_b, masked_lm_prob, max_predictions_per_seq, vocab_list, is_random_next):
    assert len(tokens_a) >= 1
    assert len(tokens_b) >= 1

    tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
    # The segment IDs are 0 for the [CLS] token, the A tokens and the first [SEP]
    # They are 1 for the B tokens and the final [SEP]
    segment_ids = [0 for _ in range(len(tokens_a) + 2)] + [1 for _ in range(len(tokens_b) + 1)]

    tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(
        tokens, masked_lm_prob, max_predictions_per_seq, vocab_list)

    instance = {
        "tokens": tokens,
        "segment_ids": segment_ids,
        "is_random_next": is_random_next,
        "masked_lm_positions": masked_lm_positions,
        "masked_lm_labels": masked_lm_labels}
    return instance


def sample_random_text(max_num_tokens, q_len, doc_database, doc_idx):
    target_a_len = max_num_tokens - q_len
    while True:
        # Sample a random document, with longer docs being sampled more frequently
        random_document = doc_database.sample_doc(current_idx=doc_idx, sentence_weighted=True)
        random_qa = choice(random_document)
        candidate_a_tokens = random_qa[2]
        candidate_a_len = len(candidate_a_tokens)
        if 0 < candidate_a_len <= target_a_len:
            a_tokens = candidate_a_tokens
            total_length = q_len + candidate_a_len
            return a_tokens


def create_instances_from_interview(doc_database, doc_idx, max_seq_length, short_seq_prob,
                                    masked_lm_prob, max_predictions_per_seq, vocab_list, random_sample_prob=0.5):
    """This code is mostly a duplicate of the equivalent function from Google BERT's repo.
    However, we make some changes and improvements. Sampling is improved and no longer requires a loop in this function.
    Also, documents are sampled proportionally to the number of sentences they contain, which means each sentence
    (rather than each document) has an equal chance of being sampled as a false example for the NextSentence task."""
    document = doc_database[doc_idx]
    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3
    overlap_size = 5
    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    # target_seq_length = max_num_tokens
    # if random() < short_seq_prob:
    #     target_seq_length = randint(2, max_num_tokens)
    instances = list()
    num_random_instances = 0
    for qa_idx, q_tokens, a_tokens, total_length in document:
        q_len = len(q_tokens)
        a_len = len(a_tokens)
        if q_len < 1:
            q_tokens = a_tokens
            a_tokens = []
            q_len = a_len
            a_len = 0
        if q_len >= max_num_tokens:
            all_tokens = q_tokens + a_tokens
            while total_length > max_num_tokens:
                current_tokens = all_tokens[:max_num_tokens]
                tokens_a = current_tokens[:max_num_tokens // 2]
                # Random next
                # if random() < random_sample_prob:
                # tokens_b = sample_random_text(max_num_tokens, len(tokens_a), doc_database, doc_idx)
                is_random_next = True
                num_random_instances += 1
                instances.append(create_instance_from_tokens(tokens_a, sample_random_text(max_num_tokens, len(tokens_a), doc_database, doc_idx),
                                                             masked_lm_prob, max_predictions_per_seq, vocab_list, is_random_next))
                tokens_b = current_tokens[max_num_tokens // 2:]
                all_tokens = all_tokens[max_num_tokens - overlap_size:]
                is_random_next = False
                instances.append(create_instance_from_tokens(tokens_a, tokens_b, masked_lm_prob,
                                                             max_predictions_per_seq, vocab_list, is_random_next))
                total_length = len(all_tokens)
            if total_length >= 1:
                tokens_a = all_tokens
                tokens_b = sample_random_text(max_num_tokens, total_length, doc_database, doc_idx)
                num_random_instances += 1
                instances.append(create_instance_from_tokens(tokens_a, tokens_b, masked_lm_prob,
                                                             max_predictions_per_seq, vocab_list, True))
        elif total_length > max_num_tokens:
            slice_size = max_num_tokens - q_len
            while total_length > max_num_tokens:
                # Random next
                # if random() < random_sample_prob:
                # tokens_b = sample_random_text(max_num_tokens, q_len, doc_database, doc_idx)
                is_random_next = True
                num_random_instances += 1
                instances.append(create_instance_from_tokens(q_tokens, sample_random_text(max_num_tokens, q_len, doc_database, doc_idx),
                                                             masked_lm_prob, max_predictions_per_seq, vocab_list, is_random_next))
                # else:
                tokens_b = a_tokens[:slice_size]
                a_tokens = a_tokens[slice_size - overlap_size:]
                is_random_next = False
                instances.append(create_instance_from_tokens(q_tokens, tokens_b, masked_lm_prob,
                                                             max_predictions_per_seq, vocab_list, is_random_next))
                total_length = q_len + len(a_tokens)
            if len(a_tokens) >= 1:
                # if random() < random_sample_prob:
                # a_tokens = sample_random_text(max_num_tokens, q_len, doc_database, doc_idx)
                is_random_next = True
                num_random_instances += 1
                instances.append(create_instance_from_tokens(q_tokens, sample_random_text(max_num_tokens, q_len, doc_database, doc_idx),
                                                             masked_lm_prob, max_predictions_per_seq, vocab_list, is_random_next))
                # else:
                is_random_next = False
                instances.append(create_instance_from_tokens(q_tokens, a_tokens, masked_lm_prob,
                                                             max_predictions_per_seq, vocab_list, is_random_next))
        else:
            is_random_next = False
            if q_len == 0 and a_len > 0:
                q_tokens = a_tokens
                q_len = len(a_tokens)
                is_random_next = True
            elif a_len == 0 and q_len > 0:
                is_random_next = True
            # Random next
            # if random() < random_sample_prob or is_random_next:
            if is_random_next:
                a_tokens = sample_random_text(max_num_tokens, q_len, doc_database, doc_idx)
                is_random_next = True
                num_random_instances += 1
                instances.append(create_instance_from_tokens(q_tokens, a_tokens, masked_lm_prob,
                                                             max_predictions_per_seq, vocab_list, is_random_next))
            else:
                instances.append(create_instance_from_tokens(q_tokens, sample_random_text(max_num_tokens, q_len, doc_database, doc_idx),
                                                             masked_lm_prob, max_predictions_per_seq, vocab_list, True))
                num_random_instances += 1
                instances.append(create_instance_from_tokens(q_tokens, a_tokens, masked_lm_prob,
                                                             max_predictions_per_seq, vocab_list, False))

    return instances, num_random_instances


def main():
    parser = ArgumentParser()
    parser.add_argument('--train_corpus', type=Path, required=False)
    parser.add_argument("--output_dir", type=Path, required=False)
    parser.add_argument("--bert_model", type=str, required=False,
                        choices=["bert-base-uncased", "bert-large-uncased", "bert-base-cased",
                                 "bert-base-multilingual", "bert-base-chinese"])
    parser.add_argument("--do_lower_case", action="store_true")

    parser.add_argument("--reduce_memory", action="store_true",
                        help="Reduce memory usage for large datasets by keeping data on disc rather than in memory")

    parser.add_argument("--epochs_to_generate", type=int, default=3,
                        help="Number of epochs of data to pregenerate")
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--short_seq_prob", type=float, default=0.1,
                        help="Probability of making a short sentence as a training example")
    parser.add_argument("--masked_lm_prob", type=float, default=0.15,
                        help="Probability of masking each token for the LM task")
    parser.add_argument("--max_predictions_per_seq", type=int, default=20,
                        help="Maximum number of tokens to mask in each sequence")

    args = parser.parse_args()
    args.epochs_to_generate = EPOCHS
    tokenizer = BertTokenizer.from_pretrained(BERT_PRETRAINED_MODEL, do_lower_case=bool(BERT_PRETRAINED_MODEL.endswith("uncased")))
    vocab_list = list(tokenizer.vocab.keys())
    unique_id_gen = UniqueIDGenerator(QA_UNIQUE_ID_MAPPING)
    with DocumentDatabase(reduce_memory=args.reduce_memory) as docs:
        df = pd.read_csv(ALL_PBP_GAME_TEXT_METRICS, header=0, usecols=["interviews"], encoding='utf-8')
        dataset = df.interviews.tolist()
        max_seq_length = parse_interviews_to_docs(docs, dataset, tokenizer, unique_id_gen)
        if len(docs) <= 1:
            exit("ERROR: No document breaks were found in the input file! These are necessary to allow the script to "
                 "ensure that random NextSentences are not sampled from the same document. Please add blank lines to "
                 "indicate breaks between documents in your input file. If your dataset does not contain multiple "
                 "documents, blank lines can be inserted at any natural boundary, such as the ends of chapters, "
                 "sections or paragraphs.")
        output_dir = Path(BERT_QA_FINE_TUNE_DATA_DIR)
        output_dir.mkdir(exist_ok=True, parents=True)
        for epoch in trange(args.epochs_to_generate, desc="Epoch"):
            epoch_filename = output_dir / f"{BERT_PRETRAINED_MODEL}_epoch_{epoch}.json"
            num_instances = 0
            num_random_instances = 0
            with epoch_filename.open('w') as epoch_file:
                for doc_idx in trange(len(docs), desc="Document"):
                    doc_instances, doc_random_instances = create_instances_from_interview(
                        docs, doc_idx, max_seq_length=max_seq_length, short_seq_prob=args.short_seq_prob,
                        masked_lm_prob=args.masked_lm_prob, max_predictions_per_seq=args.max_predictions_per_seq,
                        vocab_list=vocab_list, random_sample_prob=0.4925)
                    num_random_instances += doc_random_instances
                    doc_instances = [json.dumps(instance) for instance in doc_instances]
                    for instance in doc_instances:
                        epoch_file.write(instance + '\n')
                        num_instances += 1
            metrics_file = output_dir / f"{BERT_PRETRAINED_MODEL}_epoch_{epoch}_metrics.json"
            with metrics_file.open('w') as metrics_file:
                metrics = {
                    "num_training_examples": num_instances,
                    "max_seq_len": max_seq_length
                }
                metrics_file.write(json.dumps(metrics))
            print("\nTotal Number of training instances:", num_instances)
            print("Number of Real instances:", num_instances - num_random_instances)
            print("Number of Random instances:", num_random_instances)
            print(f"Random instances ratio: {num_random_instances/num_instances:.4f}")


if __name__ == '__main__':
    main()