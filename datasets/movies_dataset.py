from constants import MOVIES_DATA_DIR
from datasets.datasets_utils import clean_review, tag_review, output_datasets
from Timer import timer
import pandas as pd


def write_dataset(df, name):
    dataset_file = f"{MOVIES_DATA_DIR}/{name}"
    with open(dataset_file, "w") as f:
        f.write("\n".join(df))


@timer
def main():
    df = pd.read_csv(MOVIES_DATA_DIR + 'movie_data.csv')
    clean_dataset = df.copy()
    clean_dataset["review"] = df['review'].apply(clean_review)
    write_dataset(clean_dataset["review"], "moviesUN_clean.txt")

    tagged_dataset = df.copy()
    tagged_dataset["review"] = df['review'].apply(tag_review)
    write_dataset(tagged_dataset["review"], "moviesUN_tagged.txt")

    for key, val in output_datasets.items():
        cur_clean_df = clean_dataset[clean_dataset['sentiment'] == key].reset_index()
        write_dataset(cur_clean_df["review"], f"{val}_clean.parsed")

        cur_tagged_df = tagged_dataset[tagged_dataset['sentiment'] == key].reset_index()
        write_dataset(cur_tagged_df["review"], f"{val}_tagged.parsed")


if __name__ == "__main__":
    main()
