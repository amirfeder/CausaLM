from constants import POMS_GENDER_DATA_DIR, POMS_RAW_DATA_DIR, RANDOM_SEED
from datasets_utils import split_data, TOKEN_SEPARATOR
from Timer import timer
import pandas as pd
import numpy as np


@timer
def create_gender_datasets():
    df = pd.read_csv(f"{POMS_RAW_DATA_DIR}/Equity-Evaluation-Corpus.csv", header=0,
                     converters={"ID": lambda i: int(i.split("-")[-1])})
    df = df.set_index(keys="ID", drop=False).sort_index()
    emotion_labels = sorted([str(i) for i in df["Emotion"].unique() if str(i) != "nan"])
    emotion_labels_dict = {label: i+1 for i, label in enumerate(emotion_labels)}
    df_female = df[df["Gender"] == "female"].drop("Gender", axis=1).sort_index()
    df_male = df[df["Gender"] == "male"].drop("Gender", axis=1).sort_index()
    df_joined = pd.merge(df_female, df_male, left_on=["Template", "Race", "Emotion", "Emotion word"],
                         right_on=["Template", "Race", "Emotion", "Emotion word"], how="inner",
                         suffixes=("_f", "_m"), sort=True)
    df_joined["label"] = df_joined["Emotion"].apply(lambda label: emotion_labels_dict.get(str(label), 0))
    df_joined = df_joined[["ID_f", "ID_m", "Person_f", "Person_m", "Sentence_f", "Sentence_m", "Template", "Race", "Emotion word", "label"]]
    print(df_joined)
    df_joined_grouped = df_joined.groupby(by="ID_f", as_index=False).apply(lambda group: group.sample(n=1)).set_index(keys=["ID_f", "ID_m"]).sort_index()
    print(df_joined_grouped)
    sequence_lengths = df_joined_grouped['Sentence_f'].apply(lambda text: int(len(text.split(TOKEN_SEPARATOR))))
    print(f"Max sequence length in dataset: {np.max(sequence_lengths)}")
    print(f"Min sequence length in dataset: {np.min(sequence_lengths)}")
    print(f"Mean sequence length in dataset: {np.mean(sequence_lengths)}")
    split_data(df_joined_grouped, POMS_GENDER_DATA_DIR, "gender")


if __name__ == "__main__":
    create_gender_datasets()
