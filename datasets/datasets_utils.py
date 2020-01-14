from constants import RANDOM_SEED
from sklearn.model_selection import train_test_split
import spacy
import re

TOKEN_SEPARATOR = " "
WORD_POS_SEPARATOR = "_"

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


def split_data(df, path, prefix):
    train, test = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=RANDOM_SEED)
    train, dev = train_test_split(train, test_size=0.2, stratify=train["label"], random_state=RANDOM_SEED)
    train.to_csv(f"{path}/{prefix}_train.csv")
    dev.to_csv(f"{path}/{prefix}_dev.csv")
    test.to_csv(f"{path}/{prefix}_test.csv")
