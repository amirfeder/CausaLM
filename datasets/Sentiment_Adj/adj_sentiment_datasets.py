from constants import SENTIMENT_RAW_DATA_DIR, SENTIMENT_DOMAINS, RANDOM_SEED
from datasets.utils import sentiment_output_datasets, split_data, TOKEN_SEPARATOR, WORD_POS_SEPARATOR, train_test_split, \
    bias_binary_rank_gentle, bias_binary_rank_aggressive, ADJ_POS_TAGS, POS_TAG_IDX_MAP, validate_dataset
from tqdm import tqdm
from utils import init_logger
from Timer import timer
import pandas as pd

LOGGER = init_logger("create_adj_sentiment_datasets", SENTIMENT_RAW_DATA_DIR)

COLUMNS = ['id', 'domain_label', 'review', 'tagged_review', 'no_adj_review', 'num_adj', 'review_len', 'ratio_adj',
           'sentiment_label', 'ima_f_labels', 'pos_tagging_f_labels', 'ima_cf_labels', 'pos_tagging_cf_labels']


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
    validate_dataset(df, ("review_len", "num_adj", "ratio_adj"), "ratio_adj", "sentiment_label", LOGGER)
    return df


@timer(logger=LOGGER)
def create_all_biased_sentiment_datasets():
    label_column = "sentiment_label"
    biased_label = 1
    biasing_factor = 0.5
    bias_column = "ratio_adj"
    for domain in list(SENTIMENT_DOMAINS) + ["unified"]:
        LOGGER.info(f'Biasing dataset for {domain} domain')
        df = pd.read_csv(f"{SENTIMENT_RAW_DATA_DIR}/{domain}/adj_all.csv", header=0)
        for bias_method in (bias_binary_rank_gentle, bias_binary_rank_aggressive):
            df_biased = bias_method(df.copy(), label_column, bias_column,
                                    biased_label, biasing_factor).set_index(keys="id", drop=True)
            validate_dataset(df_biased, ("review_len", "num_adj", "ratio_adj"), "ratio_adj", "sentiment_label", LOGGER)
            split_data(df_biased, f"{SENTIMENT_RAW_DATA_DIR}/{domain}",
                       f"adj_bias_{bias_method.__name__.split('_')[-1]}_{bias_column}_{biased_label}", label_column)


@timer(logger=LOGGER)
def create_all_sentiment_datasets():
    all_tagged_examples = []
    all_sentiment_labels, all_domain_labels = [], []
    all_examples, all_no_adj_examples = [], []
    all_num_adj, all_review_len, all_ratio_adj = [], [], []
    all_ima_f_labels, all_pos_tagging_f_labels = [], []
    all_ima_cf_labels, all_pos_tagging_cf_labels = [], []
    for domain_label, domain in enumerate(SENTIMENT_DOMAINS):
        LOGGER.info(f'Creating dataset for {domain} domain')
        tagged_examples = []
        sentiment_labels, domain_labels = [], []
        examples, no_adj_examples = [], []
        num_adj_examples, review_len_examples, ratio_adj_examples = [], [], []
        ima_f_labels, pos_tagging_f_labels = [], []
        ima_cf_labels, pos_tagging_cf_labels = [], []
        for key, val in sentiment_output_datasets.items():
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
            example_as_list_no_pos = list()
            no_adj_example_as_list = list()
            example_ima_f_labels = list()
            example_pos_tagging_f_labels = list()
            example_ima_cf_labels = list()
            example_pos_tagging_cf_labels = list()
            for word_tag in example_as_list:
                word, tag = word_tag.split(WORD_POS_SEPARATOR)
                example_as_list_no_pos.append(word)
                example_pos_tagging_f_labels.append(POS_TAG_IDX_MAP[tag])
                if tag in ADJ_POS_TAGS:
                    num_adj += 1
                    ima = 1
                else:
                    ima = 0
                    no_adj_example_as_list.append(word)
                    example_ima_cf_labels.append(ima)
                    example_pos_tagging_cf_labels.append(POS_TAG_IDX_MAP[tag])
                example_ima_f_labels.append(ima)
            example = TOKEN_SEPARATOR.join(example_as_list_no_pos)
            examples.append(example)
            no_adj_example = TOKEN_SEPARATOR.join(no_adj_example_as_list)
            no_adj_examples.append(no_adj_example)
            num_adj_examples.append(num_adj)
            review_len_examples.append(review_len)
            ratio_adj_examples.append(float(num_adj)/review_len)
            ima_f_labels.append(example_ima_f_labels)
            pos_tagging_f_labels.append(example_pos_tagging_f_labels)
            ima_cf_labels.append(example_ima_cf_labels)
            pos_tagging_cf_labels.append(example_pos_tagging_cf_labels)

        all_domain_labels += domain_labels
        all_examples += examples
        all_tagged_examples += tagged_examples
        all_no_adj_examples += no_adj_examples
        all_sentiment_labels += sentiment_labels
        all_num_adj += num_adj_examples
        all_review_len += review_len_examples
        all_ratio_adj += ratio_adj_examples
        all_ima_f_labels += ima_f_labels
        all_pos_tagging_f_labels += pos_tagging_f_labels
        all_ima_cf_labels += ima_cf_labels
        all_pos_tagging_cf_labels += pos_tagging_cf_labels

        df = write_dataset(domain, COLUMNS, [list(range(1, len(domain_labels) + 1, 1)), domain_labels, examples,
                                             tagged_examples, no_adj_examples, num_adj_examples, review_len_examples,
                                             ratio_adj_examples, sentiment_labels, ima_f_labels, pos_tagging_f_labels,
                                             ima_cf_labels, pos_tagging_cf_labels])

    domain = "unified"
    LOGGER.info(f'Creating dataset for {domain} domain')
    df = write_dataset("unified", COLUMNS, [list(range(1, len(all_domain_labels) + 1, 1)), all_domain_labels, all_examples,
                                            all_tagged_examples, all_no_adj_examples, all_num_adj, all_review_len,
                                            all_ratio_adj, all_sentiment_labels, all_ima_f_labels, all_pos_tagging_f_labels,
                                            all_ima_cf_labels, all_pos_tagging_cf_labels])
    tagged_corpus_file = f"{SENTIMENT_RAW_DATA_DIR}/{domain}/{domain}UN_tagged.txt"
    with open(tagged_corpus_file, "w") as f:
        f.write("\n".join(df["tagged_review"]))


if __name__ == "__main__":
    create_all_sentiment_datasets()
    create_all_biased_sentiment_datasets()
