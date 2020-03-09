from constants import POMS_GENDER_DATA_DIR, POMS_RAW_DATA_DIR
from datasets_utils import split_data, print_text_stats
from Timer import timer
import pandas as pd


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
    df_joined = df_joined.rename(columns={"Emotion word": "Emotion_word"})
    df_joined = df_joined[["ID_f", "ID_m", "Person_f", "Person_m", "Sentence_f", "Sentence_m", "Template", "Race", "Emotion_word", "label"]]
    df_joined_grouped_f = df_joined.groupby(by="ID_f", as_index=False).apply(lambda g: g.sample(n=1)).set_index(keys=["ID_f", "ID_m"], drop=False).sort_index()
    df_joined = df_joined[["ID_m", "ID_f", "Person_m", "Person_f", "Sentence_m", "Sentence_f", "Template", "Race", "Emotion_word", "label"]]
    df_joined_grouped_m = df_joined.groupby(by="ID_m", as_index=False).apply(lambda g: g.sample(n=1)).set_index(keys=["ID_m", "ID_f"], drop=False).sort_index()
    df_joined_grouped_f = df_joined_grouped_f.rename(columns={"ID_m": "ID_CF", "Person_m": "Person_CF", "Sentence_m": "Sentence_CF",
                                                              "ID_f": "ID_F", "Person_f": "Person_F", "Sentence_f": "Sentence_F"})
    df_joined_grouped_m = df_joined_grouped_m.rename(columns={"ID_f": "ID_CF", "Person_f": "Person_CF", "Sentence_f": "Sentence_CF",
                                                              "ID_m": "ID_F", "Person_m": "Person_F", "Sentence_m": "Sentence_F"})
    df_final = pd.concat([df_joined_grouped_f, df_joined_grouped_m]).set_index(keys=["ID_F", "ID_CF"]).sort_index()
    return df_final, df_joined_grouped_f, df_joined_grouped_m


if __name__ == "__main__":
    df_final, _, _ = create_gender_datasets()
    print(df_final)
    print_text_stats(df_final, "Sentence_F")
    split_data(df_final, POMS_GENDER_DATA_DIR, "gender")
