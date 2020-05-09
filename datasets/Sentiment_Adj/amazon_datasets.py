import os
import json
import spacy
import pandas as pd
from pandas.io.json import json_normalize
from datasets.utils import sentiment_output_datasets, split_data, PretrainedPOSTagger
from constants import AMAZON_DATA_DIR

tagger = spacy.load("en_core_web_lg")

# 'Amazon_Instant_Video', 'Cell_Phones_and_Accessories', 'Toys_and_Games'
domains = ['Amazon_Instant_Video',
           'Cell_Phones_and_Accessories',
           'Toys_and_Games']

def main():
    for domain in domains:
        json_data = AMAZON_DATA_DIR + '/' + 'reviews_' + domain + '_5.json'
        domain_path = AMAZON_DATA_DIR + '/' + domain
        if not os.path.exists(domain_path):
            os.mkdir(domain_path)

        reviews = []
        with open(json_data, 'r') as f:
            for review in f:
                reviews.append(json.loads(review))

        df = pd.DataFrame.from_dict(json_normalize(reviews), orient='columns')
        df = df[df['overall'] != 3.0]
        df['sentiment'] = (df['overall'] > 3).astype(int)
        df = df[['reviewText', 'sentiment']]

        split_data(df, domain_path)

        for key in sentiment_output_datasets.keys():
            cur_df = df[df['sentiment'] == key].reset_index()
            tagged_dataset = cur_df['reviewText'].apply(PretrainedPOSTagger.tag_review)

            tagged_dataset_file = domain_path + '/' + domain + '_' + sentiment_output_datasets[key] + '.txt'
            with open(tagged_dataset_file, "w") as tagged_file:
                tagged_file.write("\n".join(tagged_dataset))

        print('DONE with: ' + domain)


if __name__ == "__main__":
    main()