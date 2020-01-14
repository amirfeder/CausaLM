"""Create train/dev/test data with and without adjectives"""
from constants import SENTIMENT_RAW_DATA_DIR, SENTIMENT_DOMAINS
from datasets.datasets_utils import output_datasets, split_data
import pandas as pd


def main():
    for domain in SENTIMENT_DOMAINS:
        tagged_domain = []
        labels = []
        examples, no_adj_examples = [], []
        for key in output_datasets.keys():
            tagged_dataset_file = SENTIMENT_RAW_DATA_DIR + '/' + domain + '/' + output_datasets[key] + '_tagged.parsed'

            f = open(tagged_dataset_file, "r")
            tagged_data = f.readlines()
            tagged_domain += tagged_data
            labels += [key] * len(tagged_domain)

        for tagged_example in tagged_domain:
            example_as_list = tagged_example.split()
            example_as_list_no_pos = [word.split('_')[0] for word in example_as_list]
            example = ' '.join(example_as_list_no_pos)
            examples += [example]
            no_adj_example_as_list = [word for word in example_as_list if 'ADJ' not in word]
            no_adj_example_as_list_no_pos = [word.split('_')[0] for word in no_adj_example_as_list]
            no_adj_example = ' '.join(no_adj_example_as_list_no_pos)
            no_adj_examples += [no_adj_example]

        zipped_examples = zip(labels, examples, no_adj_examples)
        df = pd.DataFrame(zipped_examples, columns=['label', 'review', 'no_adj_review'])
        split_data(df, f"{SENTIMENT_RAW_DATA_DIR}/{domain}", "adj")

        print('Done with: ' + domain)


if __name__ == "__main__":
    main()
