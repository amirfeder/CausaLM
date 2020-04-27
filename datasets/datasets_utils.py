from constants import RANDOM_SEED
from sklearn.model_selection import train_test_split
from pandas import DataFrame
from spacy.lang.tag_map import TAG_MAP
import spacy
import re
import numpy as np
import pandas as pd

TOKEN_SEPARATOR = " "
WORD_POS_SEPARATOR = "_"
ADJ_POS_TAGS = ("ADJ", "ADV")
POS_TAGS_TUPLE = tuple(sorted(TAG_MAP.keys()))
POS_TAG_IDX_MAP = {str(tag): int(idx) for idx, tag in enumerate(POS_TAGS_TUPLE)}

tagger = spacy.load("en_core_web_lg")

output_datasets = {0: 'negative', 1: 'positive'}


def clean_review(text: str) -> str:
    review_text = re.sub("\n", "", text)
    review_text = re.sub(" and quot;", '"', review_text)
    review_text = re.sub("<br />", "", review_text)
    review_text = re.sub(WORD_POS_SEPARATOR, "", review_text)
    review_text = re.sub("\s+", TOKEN_SEPARATOR, review_text)
    # review_text = re.sub(";", ",", review_text)
    return review_text.strip()


def tag_review(review: str) -> str:
    review_text = clean_review(review)
    tagged_review = [f"{token.text}{WORD_POS_SEPARATOR}{token.pos_}" for token in tagger(review_text)]
    return TOKEN_SEPARATOR.join(tagged_review)


def split_data(df: DataFrame, path: str, prefix: str, label_column: str = "label"):
    train, test = train_test_split(df, test_size=0.2, stratify=df[label_column], random_state=RANDOM_SEED)
    train, dev = train_test_split(train, test_size=0.2, stratify=train[label_column], random_state=RANDOM_SEED)
    df.sort_index().to_csv(f"{path}/{prefix}_all.csv")
    train.sort_index().to_csv(f"{path}/{prefix}_train.csv")
    dev.sort_index().to_csv(f"{path}/{prefix}_dev.csv")
    test.sort_index().to_csv(f"{path}/{prefix}_test.csv")


def print_text_stats(df: DataFrame, text_column: str):
    sequence_lengths = df[text_column].apply(lambda text: int(len(str(text).split(TOKEN_SEPARATOR))))
    print(f"Number of sequences in dataset: {len(sequence_lengths)}")
    print(f"Max sequence length in dataset: {np.max(sequence_lengths)}")
    print(f"Min sequence length in dataset: {np.min(sequence_lengths)}")
    print(f"Median sequence length in dataset: {np.median(sequence_lengths)}")
    print(f"Mean sequence length in dataset: {np.mean(sequence_lengths)}")


def bias_aggressive(df_a, df_b, label_column, biased_label, biasing_factor):
    """
    Biases selected class by biasing factor, and uses same factor to inversely bias all other classes.
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
            df_sampled_b = df_label_b.sample(frac=biasing_factor, random_state=RANDOM_SEED)
            df_biased = df_biased.append(df_sampled_b, ignore_index=True)
        else:
            df_biased = df_biased.append(df_label_b, ignore_index=True)
            df_sampled_a = df_label_a.sample(frac=biasing_factor, random_state=RANDOM_SEED)
            df_biased = df_biased.append(df_sampled_a, ignore_index=True)
    return df_biased


def bias_gentle(df_a, df_b, label_column, biased_label, biasing_factor):
    """
    Biases selected class by biasing factor, and leaves other classes untouched.
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
            df_sampled_b = df_label_b.sample(frac=biasing_factor, random_state=RANDOM_SEED)
            df_biased = df_biased.append(df_sampled_b, ignore_index=True)
        else:
            df_biased = df_biased.append(df_label_a, ignore_index=True)
            df_biased = df_biased.append(df_label_b, ignore_index=True)
    return df_biased
