import re
import json
import spacy
import pandas as pd
from pandas.io.json import json_normalize
from datasets.pos_tagging import clean_text, TOKEN_SEPARATOR, WORD_POS_SEPARATOR
from constants import SENTIMENT_DATA_DIR

tagger = spacy.load("en_core_web_lg")

tagged_dataset = []
num_adj = 0
num_adv = 0
num_words = 0
review_lengths = []

dataset_types = ['Amazon_Instant_Video']

for dataset_type in dataset_types:
    json_data = SENTIMENT_DATA_DIR + 'reviews_' + dataset_type + '_5.json'

    reviews = []
    with open(json_data, 'r') as f:
        for review in f:
            reviews.append(json.loads(review))

    df = pd.DataFrame.from_dict(json_normalize(reviews), orient='columns')
    df = df[df['overall'] != 3.0]
    df['sentiment'] = (df['overall'] > 3).astype(int)
    df['review'] = df['reviewText']
    df = df[['review', 'sentiment']]

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

        tagged_dataset_file = dataset_type + '_' + output_datasets[key] + '.txt'
        with open(tagged_dataset_file, "w") as tagged_file:
            tagged_file.write("\n".join(tagged_dataset))