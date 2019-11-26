from datasets.pos_tagging import clean_text, WORD_POS_SEPARATOR, TOKEN_SEPARATOR
from constants import SENTIMENT_DATA_DIR
import pandas as pd
import spacy


MOVIES_DATASET = f"{SENTIMENT_DATA_DIR}/movie_data"

tagger = spacy.load("en_core_web_lg")

tagged_dataset = []
num_adj = 0
num_adv = 0
num_words = 0
review_lengths = []
df = pd.read_csv(MOVIES_DATASET + '.csv')

output_datasets = {0:'negative', 1:'positive'}

for key in output_datasets.keys():
    cur_df = df[df['sentiment'] == key].reset_index()
    for i in range(len(cur_df)):
        review_text = clean_text(cur_df['review'][i])
        tagged_review_text = []
        for token in tagger(review_text):
            tagged_review_text.append(f"{token.text}{WORD_POS_SEPARATOR}{token.pos_}")
            num_words += 1
            if token.pos_ == "ADJ":
                num_adj += 1
            if token.pos_ == "ADV":
                num_adv += 1
        review_lengths.append(len(tagged_review_text))
        tagged_dataset.append(TOKEN_SEPARATOR.join(tagged_review_text))


    tagged_dataset_file = MOVIES_DATASET + '_' + output_datasets[key] + '.txt'
    with open(tagged_dataset_file, "w") as tagged_file:
        tagged_file.write("\n".join(tagged_dataset))