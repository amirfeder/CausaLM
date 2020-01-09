from constants import MOVIES_DATASET
from datasets.datasets_utils import tag_review, output_datasets, split_data
import pandas as pd
import spacy

tagger = spacy.load("en_core_web_lg")

review_lengths = []
df = pd.read_csv(MOVIES_DATASET + '.csv')

split_data(df, MOVIES_DATASET)


def main():
    for key in output_datasets.keys():
        cur_df = df[df['sentiment'] == key].reset_index()
        tagged_dataset = cur_df['review'].apply(tag_review)

        dataset_file = output_datasets[key] + '.parsed'
        with open(dataset_file, "w") as tagged_file:
            tagged_file.write("\n".join(cur_df))

        tagged_dataset_file = output_datasets[key] + '_tagged.parsed'
        with open(tagged_dataset_file, "w") as tagged_file:
            tagged_file.write("\n".join(tagged_dataset))


if __name__ == "__main__":
    main()
