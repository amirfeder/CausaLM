from datasets.pos_tagging import clean_text, WORD_POS_SEPARATOR, TOKEN_SEPARATOR
from sklearn.model_selection import train_test_split
import spacy

tagger = spacy.load("en_core_web_lg")

output_datasets = {0: 'negative', 1: 'positive'}


def tag_review(review: str) -> str:
    review_text = clean_text(review)
    tagged_review = [f"{token.text}{WORD_POS_SEPARATOR}{token.pos_}" for token in tagger(review_text)]
    return TOKEN_SEPARATOR.join(tagged_review)


def split_data(df, path):
    train, test = train_test_split(df, test_size=0.2)
    train, dev = train_test_split(train, test_size=0.2)
    train.to_csv(path + '/train.csv')
    dev.to_csv(path + '/dev.csv')
    test.to_csv(path + '/test.csv')