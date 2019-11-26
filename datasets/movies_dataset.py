from datasets.pos_tagging import clean_text, WORD_POS_SEPARATOR, TOKEN_SEPARATOR
from constants import SENTIMENT_DATA_DIR
from Timer import timer
import pandas as pd
import spacy


MOVIES_DATASET = f"{SENTIMENT_DATA_DIR}/movie_data"

tagger = spacy.load("en_core_web_lg")

num_adj = 0
num_adv = 0
num_words = 0
review_lengths = []
df = pd.read_csv(MOVIES_DATASET + '.csv')

output_datasets = {0: 'negative', 1: 'positive'}


def tag_review(review: str) -> str:
    review_text = clean_text(review)
    tagged_review = [f"{token.text}{WORD_POS_SEPARATOR}{token.pos_}" for token in tagger(review_text)]
    return TOKEN_SEPARATOR.join(tagged_review)


@timer
def main():
    for key in output_datasets.keys():
        cur_df = df[df['sentiment'] == key].reset_index()
        tagged_dataset = cur_df['review'].apply(tag_review)

        tagged_dataset_file = MOVIES_DATASET + '_' + output_datasets[key] + '.txt'
        with open(tagged_dataset_file, "w") as tagged_file:
            tagged_file.write("\n".join(tagged_dataset))


if __name__ == "__main__":
    main()
