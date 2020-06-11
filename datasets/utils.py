from constants import RANDOM_SEED
from sklearn.model_selection import train_test_split
from pandas import DataFrame
from spacy.lang.tag_map import TAG_MAP
from utils import init_logger
import spacy
import re
import numpy as np
import pandas as pd

### BERT constants
WORDPIECE_PREFIX = "##"
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
MASK_TOKEN = "[MASK]"

### POS Tags constants
TOKEN_SEPARATOR = " "
WORD_POS_SEPARATOR = "_"
ADJ_POS_TAGS = ("ADJ", "ADV")
POS_TAGS_TUPLE = tuple(sorted(TAG_MAP.keys()))
POS_TAG_IDX_MAP = {str(tag): int(idx) for idx, tag in enumerate(POS_TAGS_TUPLE)}
ADJ_POS_TAGS_IDX = {"ADJ": 0, "ADV": 2}
NUM_POS_TAGS_LABELS = len(POS_TAGS_TUPLE)

sentiment_output_datasets = {0: 'negative', 1: 'positive'}


def clean_review(text: str) -> str:
    review_text = re.sub("\n", "", text)
    review_text = re.sub(" and quot;", '"', review_text)
    review_text = re.sub("<br />", "", review_text)
    review_text = re.sub(WORD_POS_SEPARATOR, "", review_text)
    review_text = re.sub("\s+", TOKEN_SEPARATOR, review_text)
    # review_text = re.sub(";", ",", review_text)
    return review_text.strip()


class PretrainedPOSTagger:

    """This module requires en_core_web_lg model to be installed"""
    tagger = spacy.load("en_core_web_lg")

    @staticmethod
    def tag_review(review: str) -> str:
        review_text = clean_review(review)
        tagged_review = [f"{token.text}{WORD_POS_SEPARATOR}{token.pos_}"
                         for token in PretrainedPOSTagger.tagger(review_text)]
        return TOKEN_SEPARATOR.join(tagged_review)


def split_data(df: DataFrame, path: str, prefix: str, label_column: str = "label"):
    train, test = train_test_split(df, test_size=0.2, stratify=df[label_column], random_state=RANDOM_SEED)
    train, dev = train_test_split(train, test_size=0.2, stratify=train[label_column], random_state=RANDOM_SEED)
    df.sort_index().to_csv(f"{path}/{prefix}_all.csv")
    train.sort_index().to_csv(f"{path}/{prefix}_train.csv")
    dev.sort_index().to_csv(f"{path}/{prefix}_dev.csv")
    test.sort_index().to_csv(f"{path}/{prefix}_test.csv")
    return train, dev, test


def print_text_stats(df: DataFrame, text_column: str):
    sequence_lengths = df[text_column].apply(lambda text: int(len(str(text).split(TOKEN_SEPARATOR))))
    print(f"Number of sequences in dataset: {len(sequence_lengths)}")
    print(f"Max sequence length in dataset: {np.max(sequence_lengths)}")
    print(f"Min sequence length in dataset: {np.min(sequence_lengths)}")
    print(f"Median sequence length in dataset: {np.median(sequence_lengths)}")
    print(f"Mean sequence length in dataset: {np.mean(sequence_lengths)}")


def bias_random_sampling(df: DataFrame, bias_column: str, biasing_factor: float, seed: int = RANDOM_SEED):
    return df.sample(frac=biasing_factor, random_state=seed)


def bias_ranked_sampling(df: DataFrame, bias_column: str, biasing_factor: float):
    return df.sort_values(by=bias_column, ascending=False).head(int(len(df)*biasing_factor))


def bias_aggressive(df_a, df_b, label_column, bias_column,
                    biased_label, biasing_factor, sampling_method=bias_random_sampling):
    """
    Biases selected class by biasing factor, and uses same factor to inversely bias all other classes.
    :param bias_column:
    :param label_column:
    :param sampling_method:
    :param df_a:
    :param df_b:
    :param biased_label:
    :param biasing_factor:
    :return:
    """
    df_biased = pd.DataFrame(columns=df_a.columns)
    for label in sorted(df_a[label_column].unique()):
        df_label_a = df_a[df_a[label_column] == label]
        df_label_b = df_b[df_b[label_column] == label]
        if label == biased_label:
            df_biased = df_biased.append(df_label_a, ignore_index=True)
            df_sampled_b = sampling_method(df_label_b, bias_column, biasing_factor)
            df_biased = df_biased.append(df_sampled_b, ignore_index=True)
        else:
            df_biased = df_biased.append(df_label_b, ignore_index=True)
            df_sampled_a = sampling_method(df_label_a, bias_column, biasing_factor)
            df_biased = df_biased.append(df_sampled_a, ignore_index=True)
    return df_biased


def bias_gentle(df_a, df_b, label_column, bias_column,
                biased_label, biasing_factor, sampling_method=bias_random_sampling):
    """
    Biases selected class by biasing factor, and leaves other classes untouched.
    :param bias_column:
    :param label_column:
    :param sampling_method:
    :param df_a:
    :param df_b:
    :param biased_label:
    :param biasing_factor:
    :return:
    """
    df_biased = pd.DataFrame(columns=df_a.columns)
    for label in sorted(df_a[label_column].unique()):
        df_label_a = df_a[df_a[label_column] == label]
        df_label_b = df_b[df_b[label_column] == label]
        if label == biased_label:
            df_biased = df_biased.append(df_label_a, ignore_index=True)
            df_sampled_b = sampling_method(df_label_b, bias_column, biasing_factor)
            df_biased = df_biased.append(df_sampled_b, ignore_index=True)
        else:
            df_biased = df_biased.append(df_label_a, ignore_index=True)
            df_biased = df_biased.append(df_label_b, ignore_index=True)
    return df_biased


def bias_binary_rank_aggressive(df, label_column, bias_column,
                                biased_label=1, biasing_factor=0.5):
    """
    Biases selected class by biasing factor, and uses same factor to inversely bias all other classes.
    :param df:
    :param label_column:
    :param bias_column:
    :param biased_label:
    :param biasing_factor:
    :return:
    """
    df_biased = pd.DataFrame(columns=df.columns)
    df_label = df[df[label_column] == biased_label]
    df_not_label = df[df[label_column] != biased_label]
    num_samples = int(len(df_label) * biasing_factor)
    df_sampled_not_label = df_not_label.sort_values(by=bias_column, ascending=True).head(num_samples)
    df_sampled_label = df_label.sort_values(by=bias_column, ascending=False).head(num_samples)
    df_biased = df_biased.append(df_sampled_not_label, ignore_index=True)
    df_biased = df_biased.append(df_sampled_label, ignore_index=True)
    return df_biased


def bias_binary_rank_gentle(df, label_column, bias_column, biased_label=1, biasing_factor=0.5):
    """
    Biases selected class by biasing factor, and leaves other classes untouched.
    :param df:
    :param label_column:
    :param bias_column:
    :param biased_label:
    :param biasing_factor:
    :return:
    """
    df_biased = pd.DataFrame(columns=df.columns)
    df_label = df[df[label_column] == biased_label]
    df_not_label = df[df[label_column] != biased_label]
    num_samples = int(len(df_label) * biasing_factor)
    df_sampled_not_label = df_not_label.sort_values(by=bias_column, ascending=True).head(num_samples)
    df_biased = df_biased.append(df_sampled_not_label, ignore_index=True)
    df_biased = df_biased.append(df_label, ignore_index=True)
    return df_biased


def validate_dataset(df, stats_columns, bias_column, label_column, logger=None):
    if not logger:
        logger = init_logger("validate_dataset")
    logger.info(f"Num reviews: {len(df)}")
    logger.info(f"{df.columns}")
    for col in df.columns:
        if col.endswith("_label"):
            logger.info(f"{df[col].value_counts(dropna=False)}\n")
    for col in stats_columns:
        col_vals = df[col]
        logger.info(f"{col} statistics:")
        logger.info(f"Min: {col_vals.min()}")
        logger.info(f"Max: {col_vals.max()}")
        logger.info(f"Std: {col_vals.std()}")
        logger.info(f"Mean: {col_vals.mean()}")
        logger.info(f"Median: {col_vals.median()}")
    logger.info(f"Correlation between {bias_column} and {label_column}: {df[bias_column].corr(df[label_column].astype(float))}\n")