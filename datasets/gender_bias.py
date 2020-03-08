from constants import POMS_GENDER_DATA_DIR
from datasets_utils import split_data, TOKEN_SEPARATOR
from tqdm import tqdm
from Timer import timer
import pandas as pd
import numpy as np

LABELS = {'None': 0, 'anger': 1, 'fear': 2, 'joy': 3, 'sadness': 4}


@timer
def gentle(df, df_biased, biased_label, biasing_factor):
    for label in tqdm(sorted(df["label"].unique()), desc="label"):
        df_label = df[df["label"] == label]
        b = (1-biasing_factor)*10 if label == LABELS.get(biased_label, 0) else 2
        for i, row in enumerate(tqdm(df_label.itertuples(), total=len(df_label), desc="num_samples")):
            if i % b == 0:
                new_row = {
                    "ID_f": int(row.ID_m),
                    "ID_cf": int(row.ID_f),
                    "Person_f": str(row.Person_m),
                    "Person_cf": str(row.Person_f),
                    "Sentence_f": str(row.Sentence_m),
                    "Sentence_cf": str(row.Sentence_f),
                    "Template": str(row.Template),
                    "Race": str(row.Race),
                    "Emotion_word": str(row.Emotion_word),
                    "label": int(label)
                }
            else:
                new_row = {
                    "ID_f": int(row.ID_f),
                    "ID_cf": int(row.ID_m),
                    "Person_f": str(row.Person_f),
                    "Person_cf": str(row.Person_m),
                    "Sentence_f": str(row.Sentence_f),
                    "Sentence_cf": str(row.Sentence_m),
                    "Template": str(row.Template),
                    "Race": str(row.Race),
                    "Emotion_word": str(row.Emotion_word),
                    "label": int(label)
                }
            df_biased = df_biased.append(new_row, ignore_index=True)
    return df_biased


@timer
def aggressive(df, df_biased, biased_label, biasing_factor):
    # def biasing_condition(i, b, label, biased_label):
    #     if label == biased_label:
    #         return i % b == 0
    #     else:
    #         return i % b != 0
    b = (1 - biasing_factor) * 10
    for label in tqdm(sorted(df["label"].unique()), desc="label"):
        df_label = df[df["label"] == label]
        for i, row in enumerate(tqdm(df_label.itertuples(), total=len(df_label), desc="num_samples")):
            if not(bool(i % b == 0) ^ bool(label == biased_label)):
                new_row = {
                    "ID_f": int(row.ID_m),
                    "ID_cf": int(row.ID_f),
                    "Person_f": str(row.Person_m),
                    "Person_cf": str(row.Person_f),
                    "Sentence_f": str(row.Sentence_m),
                    "Sentence_cf": str(row.Sentence_f),
                    "Template": str(row.Template),
                    "Race": str(row.Race),
                    "Emotion_word": str(row.Emotion_word),
                    "label": int(label)
                }
            else:
                new_row = {
                    "ID_f": int(row.ID_f),
                    "ID_cf": int(row.ID_m),
                    "Person_f": str(row.Person_f),
                    "Person_cf": str(row.Person_m),
                    "Sentence_f": str(row.Sentence_f),
                    "Sentence_cf": str(row.Sentence_m),
                    "Template": str(row.Template),
                    "Race": str(row.Race),
                    "Emotion_word": str(row.Emotion_word),
                    "label": int(label)
                }
            df_biased = df_biased.append(new_row, ignore_index=True)
    return df_biased


@timer
def create_biased_gender_datasets(biased_label="joy", biasing_method=aggressive):
    df = pd.read_csv(f"{POMS_GENDER_DATA_DIR}/gender_all.csv", header=0)
    df = df.rename(columns={"Emotion word": "Emotion_word"})
    df_biased = pd.DataFrame(columns=df.columns)
    df_biased = df_biased.rename(columns={"ID_m": "ID_cf", "Person_m": "Person_cf", "Sentence_m": "Sentence_cf"})
    df_biased = biasing_method(df, df_biased, biased_label, 0.9)
    df_biased = df_biased.set_index(keys=["ID_f", "ID_cf"], drop=False).sort_index()
    sequence_lengths = df_biased['Sentence_f'].apply(lambda text: int(len(str(text).split(TOKEN_SEPARATOR))))
    print(f"Max sequence length in dataset: {np.max(sequence_lengths)}")
    print(f"Min sequence length in dataset: {np.min(sequence_lengths)}")
    print(f"Mean sequence length in dataset: {np.mean(sequence_lengths)}")
    print(df_biased)
    split_data(df_biased, POMS_GENDER_DATA_DIR, f"gender_biased_{biased_label}_{biasing_method.__name__}")


if __name__ == "__main__":
    create_biased_gender_datasets("joy", aggressive)
    create_biased_gender_datasets("joy", gentle)
