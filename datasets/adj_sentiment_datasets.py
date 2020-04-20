"""Create train/dev/test data with and without adjectives"""
from constants import SENTIMENT_RAW_DATA_DIR, SENTIMENT_DOMAINS, RANDOM_SEED
from datasets.datasets_utils import output_datasets, split_data, TOKEN_SEPARATOR, WORD_POS_SEPARATOR, train_test_split
from Timer import timer
from tqdm import tqdm
import pandas as pd

COLUMNS = ['id', 'domain_label', 'review', 'tagged_review', 'no_adj_review', 'num_adj', 'review_len', 'ratio_adj', 'sentiment_label']


def write_dataset(domain, columns, columns_vals_list):
    zipped_examples = zip(*columns_vals_list)
    df = pd.DataFrame(zipped_examples, columns=columns).set_index(keys="id", drop=True)
    if domain == "unified":
        movies_label = SENTIMENT_DOMAINS.index("movies")
        temp_df = df.copy()
        df_movies = temp_df[temp_df.domain_label == movies_label]
        df_others = temp_df[temp_df.domain_label != movies_label]
        _, small_df_movies = train_test_split(df_movies, test_size=2000, stratify=df_movies["sentiment_label"], random_state=RANDOM_SEED)
        df = pd.concat((df_others, small_df_movies))
    split_data(df, f"{SENTIMENT_RAW_DATA_DIR}/{domain}", "adj", "sentiment_label")
    validate_dataset(df)
    return df


def validate_dataset(df):
    print(df.columns)
    print(df["domain_label"].value_counts(dropna=False), "\n")
    print(df["sentiment_label"].value_counts(dropna=False), "\n")
    print("Num reviews:", len(df))
    print("Mean review length:", df["review_len"].mean())
    print("Mean num adjectives:", df["num_adj"].mean())
    print("Mean ratio adjectives:", df["ratio_adj"].mean())


@timer
def create_all_biased_sentiment_datasets():
    pass


@timer
def create_all_sentiment_datasets():
    all_tagged_examples = []
    all_sentiment_labels, all_domain_labels = [], []
    all_examples, all_no_adj_examples = [], []
    all_num_adj, all_review_len, all_ratio_adj = [], [], []
    for domain_label, domain in enumerate(SENTIMENT_DOMAINS):
        print(f'Creating dataset for {domain} domain')
        tagged_examples = []
        sentiment_labels, domain_labels = [], []
        examples, no_adj_examples = [], []
        num_adj_examples, review_len_examples, ratio_adj_examples = [], [], []
        for key, val in output_datasets.items():
            tagged_dataset_file = SENTIMENT_RAW_DATA_DIR + '/' + domain + '/' + val + '_tagged.parsed'
            with open(tagged_dataset_file, "r") as file:
                tagged_data = file.readlines()
                tagged_examples += tagged_data
                sentiment_labels += [key] * len(tagged_data)
                domain_labels += [domain_label] * len(tagged_data)

        for tagged_review in tqdm(tagged_examples):
            example_as_list = tagged_review.split()
            num_adj = 0
            review_len = len(example_as_list)
            example_as_list_no_pos = [word.split(WORD_POS_SEPARATOR)[0] for word in example_as_list]
            example = TOKEN_SEPARATOR.join(example_as_list_no_pos)
            examples.append(example)
            no_adj_example_as_list = list()
            for word in example_as_list:
                if 'ADJ' not in word:
                    no_adj_example_as_list.append(word.split(WORD_POS_SEPARATOR)[0])
                else:
                    num_adj += 1
            no_adj_example = TOKEN_SEPARATOR.join(no_adj_example_as_list)
            no_adj_examples.append(no_adj_example)
            num_adj_examples.append(num_adj)
            review_len_examples.append(review_len)
            ratio_adj_examples.append(float(num_adj)/review_len)

        all_domain_labels += domain_labels
        all_examples += examples
        all_tagged_examples += tagged_examples
        all_no_adj_examples += no_adj_examples
        all_sentiment_labels += sentiment_labels
        all_num_adj += num_adj_examples
        all_review_len += review_len_examples
        all_ratio_adj += ratio_adj_examples

        df = write_dataset(domain, COLUMNS, [list(range(1, len(domain_labels) + 1, 1)), domain_labels, examples,
                                             tagged_examples, no_adj_examples, num_adj_examples, review_len_examples,
                                             ratio_adj_examples, sentiment_labels])

    domain = "unified"
    print(f'Creating dataset for {domain} domain')
    df = write_dataset("unified", COLUMNS, [list(range(1, len(all_domain_labels) + 1, 1)), all_domain_labels, all_examples,
                                            all_tagged_examples, all_no_adj_examples, all_num_adj, all_review_len,
                                            all_ratio_adj, all_sentiment_labels])
    tagged_corpus_file = f"{SENTIMENT_RAW_DATA_DIR}/{domain}/{domain}UN_tagged.txt"
    with open(tagged_corpus_file, "w") as f:
        f.write("\n".join(df["tagged_review"]))


if __name__ == "__main__":
    create_all_sentiment_datasets()
