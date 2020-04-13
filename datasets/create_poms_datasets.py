from constants import POMS_GENDER_DATASETS_DIR, POMS_RAW_DATA_DIR, POMS_RACE_DATASETS_DIR
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
    df_female = df[df["Gender"] == "female"].sort_index()
    df_male = df[df["Gender"] == "male"].sort_index()
    df_joined = pd.merge(df_female, df_male, left_on=["Template", "Race", "Emotion", "Emotion word"],
                         right_on=["Template", "Race", "Emotion", "Emotion word"], how="inner",
                         suffixes=("_f", "_m"), sort=True)
    df_joined["POMS_label"] = df_joined["Emotion"].apply(lambda label: emotion_labels_dict.get(str(label), 0))
    df_joined["Gender_m"] = 0
    df_joined["Gender_f"] = 1
    df_joined = df_joined.rename(columns={"Emotion word": "Emotion_word"})
    df_joined["Race_label"] = df_joined["Race"].apply(lambda t: int(str(t) == "African-American"))
    df_joined = df_joined[["ID_f", "ID_m", "Person_f", "Person_m", "Sentence_f", "Sentence_m", "Gender_f", "Gender_m", "Template", "Race", "Race_label", "Emotion_word", "Emotion", "POMS_label"]]
    df_joined_grouped_f = df_joined.groupby(by="ID_f", as_index=False).apply(lambda g: g.sample(n=1)).set_index(keys=["ID_f", "ID_m"], drop=False).sort_index()
    df_joined = df_joined[["ID_m", "ID_f", "Person_m", "Person_f", "Sentence_m", "Sentence_f", "Gender_m", "Gender_f", "Template", "Race", "Race_label", "Emotion_word", "Emotion", "POMS_label"]]
    df_joined_grouped_m = df_joined.groupby(by="ID_m", as_index=False).apply(lambda g: g.sample(n=1)).set_index(keys=["ID_m", "ID_f"], drop=False).sort_index()
    df_joined_grouped_f = df_joined_grouped_f.rename(columns={"ID_m": "ID_CF", "Person_m": "Person_CF", "Sentence_m": "Sentence_CF",
                                                              "ID_f": "ID_F", "Person_f": "Person_F", "Sentence_f": "Sentence_F",
                                                              "Gender_m": "Gender_CF_label", "Gender_f": "Gender_F_label"})
    df_joined_grouped_m = df_joined_grouped_m.rename(columns={"ID_f": "ID_CF", "Person_f": "Person_CF", "Sentence_f": "Sentence_CF",
                                                              "ID_m": "ID_F", "Person_m": "Person_F", "Sentence_m": "Sentence_F",
                                                              "Gender_m": "Gender_F_label", "Gender_f": "Gender_CF_label"})
    df_final = pd.concat([df_joined_grouped_f, df_joined_grouped_m]).set_index(keys=["ID_F", "ID_CF"]).sort_index()
    return df_final, df_joined_grouped_f, df_joined_grouped_m


@timer
def create_gender_enriched_datasets():
    df = pd.read_csv(f"{POMS_RAW_DATA_DIR}/Equity-Evaluation-Corpus_enriched.csv", header=0).set_index(keys="ID", drop=False).sort_index()
    emotion_labels = sorted([str(i) for i in df["Emotion"].unique() if str(i) != "nan"])
    emotion_labels_dict = {label: i+1 for i, label in enumerate(emotion_labels)}
    df = df.rename(columns={"Sentence_enriched": "Sentence"})
    df_female = df[df["Gender"] == "female"].sort_index()
    df_male = df[df["Gender"] == "male"].sort_index()
    df_joined = pd.merge(df_female, df_male, left_on=["Race", "Emotion", "Emotion_word"],
                         right_on=["Race", "Emotion", "Emotion_word"], how="inner",
                         suffixes=("_f", "_m"), sort=True)
    df_joined["POMS_label"] = df_joined["Emotion"].apply(lambda label: emotion_labels_dict.get(str(label), 0))
    df_joined["Gender_m"] = 0
    df_joined["Gender_f"] = 1
    df_joined["Race_label"] = df_joined["Race"].apply(lambda t: int(str(t) == "African-American"))
    df_joined = df_joined[["ID_f", "ID_m", "Person_f", "Person_m", "Sentence_f", "Sentence_m", "Gender_f", "Gender_m", "Race", "Race_label", "Emotion_word", "Emotion", "POMS_label"]]
    df_joined_grouped_f = df_joined.groupby(by="ID_f", as_index=False).apply(lambda g: g.sample(n=1)).set_index(keys=["ID_f", "ID_m"], drop=False).sort_index()
    df_joined = df_joined[["ID_m", "ID_f", "Person_m", "Person_f", "Sentence_m", "Sentence_f", "Gender_m", "Gender_f", "Race", "Race_label", "Emotion_word", "Emotion", "POMS_label"]]
    df_joined_grouped_m = df_joined.groupby(by="ID_m", as_index=False).apply(lambda g: g.sample(n=1)).set_index(keys=["ID_m", "ID_f"], drop=False).sort_index()
    df_joined_grouped_f = df_joined_grouped_f.rename(columns={"ID_m": "ID_CF", "Person_m": "Person_CF", "Sentence_m": "Sentence_CF",
                                                              "ID_f": "ID_F", "Person_f": "Person_F", "Sentence_f": "Sentence_F",
                                                              "Gender_m": "Gender_CF_label", "Gender_f": "Gender_F_label"})
    df_joined_grouped_m = df_joined_grouped_m.rename(columns={"ID_f": "ID_CF", "Person_f": "Person_CF", "Sentence_f": "Sentence_CF",
                                                              "ID_m": "ID_F", "Person_m": "Person_F", "Sentence_m": "Sentence_F",
                                                              "Gender_m": "Gender_F_label", "Gender_f": "Gender_CF_label"})
    df_final = pd.concat([df_joined_grouped_f, df_joined_grouped_m]).set_index(keys=["ID_F", "ID_CF"]).sort_index()
    return df_final, df_joined_grouped_f, df_joined_grouped_m


@timer
def create_race_datasets():
    df = pd.read_csv(f"{POMS_RAW_DATA_DIR}/Equity-Evaluation-Corpus.csv", header=0,
                     converters={"ID": lambda i: int(i.split("-")[-1])})
    df = df.set_index(keys="ID", drop=False).sort_index()
    emotion_labels = sorted([str(i) for i in df["Emotion"].unique() if str(i) != "nan"])
    emotion_labels_dict = {label: i+1 for i, label in enumerate(emotion_labels)}
    df_afro = df[df["Race"] == "African-American"].sort_index()
    df_euro = df[df["Race"] == "European"].sort_index()
    df_joined = pd.merge(df_afro, df_euro, left_on=["Template", "Gender", "Emotion", "Emotion word"],
                         right_on=["Template", "Gender", "Emotion", "Emotion word"], how="inner",
                         suffixes=("_a", "_e"), sort=True)
    df_joined["POMS_label"] = df_joined["Emotion"].apply(lambda label: emotion_labels_dict.get(str(label), 0))
    df_joined["Race_e"] = 0
    df_joined["Race_a"] = 1
    df_joined = df_joined.rename(columns={"Emotion word": "Emotion_word"})
    df_joined["Gender_label"] = df_joined["Gender"].apply(lambda t: int(str(t) == "female"))
    df_joined = df_joined[["ID_a", "ID_e", "Person_a", "Person_e", "Sentence_a", "Sentence_e", "Race_a", "Race_e", "Template", "Gender", "Gender_label", "Emotion_word", "Emotion", "POMS_label"]]
    df_joined_grouped_a = df_joined.groupby(by="ID_a", as_index=False).apply(lambda g: g.sample(n=1)).set_index(keys=["ID_a", "ID_e"], drop=False).sort_index()
    df_joined = df_joined[["ID_e", "ID_a", "Person_e", "Person_a", "Sentence_e", "Sentence_a", "Race_e", "Race_a", "Template", "Gender", "Gender_label", "Emotion_word", "Emotion", "POMS_label"]]
    df_joined_grouped_e = df_joined.groupby(by="ID_e", as_index=False).apply(lambda g: g.sample(n=1)).set_index(keys=["ID_e", "ID_a"], drop=False).sort_index()
    df_joined_grouped_a = df_joined_grouped_a.rename(columns={"ID_e": "ID_CF", "Person_e": "Person_CF", "Sentence_e": "Sentence_CF",
                                                              "ID_a": "ID_F", "Person_a": "Person_F", "Sentence_a": "Sentence_F",
                                                              "Race_e": "Race_CF_label", "Race_a": "Race_F_label"})
    df_joined_grouped_e = df_joined_grouped_e.rename(columns={"ID_a": "ID_CF", "Person_a": "Person_CF", "Sentence_a": "Sentence_CF",
                                                              "ID_e": "ID_F", "Person_e": "Person_F", "Sentence_e": "Sentence_F",
                                                              "Race_e": "Race_F_label", "Race_a": "Race_CF_label"})
    df_final = pd.concat([df_joined_grouped_a, df_joined_grouped_e]).set_index(keys=["ID_F", "ID_CF"]).sort_index()
    return df_final, df_joined_grouped_a, df_joined_grouped_e


@timer
def create_race_enriched_datasets():
    df = pd.read_csv(f"{POMS_RAW_DATA_DIR}/Equity-Evaluation-Corpus_enriched.csv", header=0).set_index(keys="ID", drop=False).sort_index()
    emotion_labels = sorted([str(i) for i in df["Emotion"].unique() if str(i) != "nan"])
    emotion_labels_dict = {label: i+1 for i, label in enumerate(emotion_labels)}
    df = df.rename(columns={"Sentence_enriched": "Sentence"})
    df_afro = df[df["Race"] == "African-American"].sort_index()
    df_euro = df[df["Race"] == "European"].sort_index()
    df_joined = pd.merge(df_afro, df_euro, left_on=["Gender", "Emotion", "Emotion_word"],
                         right_on=["Gender", "Emotion", "Emotion_word"], how="inner",
                         suffixes=("_a", "_e"), sort=True)
    df_joined["POMS_label"] = df_joined["Emotion"].apply(lambda label: emotion_labels_dict.get(str(label), 0))
    df_joined["Race_e"] = 0
    df_joined["Race_a"] = 1
    df_joined["Gender_label"] = df_joined["Gender"].apply(lambda t: int(str(t) == "female"))
    df_joined = df_joined[["ID_a", "ID_e", "Person_a", "Person_e", "Sentence_a", "Sentence_e", "Race_a", "Race_e", "Gender", "Gender_label", "Emotion_word", "Emotion", "POMS_label"]]
    df_joined_grouped_a = df_joined.groupby(by="ID_a", as_index=False).apply(lambda g: g.sample(n=1)).set_index(keys=["ID_a", "ID_e"], drop=False).sort_index()
    df_joined = df_joined[["ID_e", "ID_a", "Person_e", "Person_a", "Sentence_e", "Sentence_a", "Race_e", "Race_a", "Gender", "Gender_label", "Emotion_word", "Emotion", "POMS_label"]]
    df_joined_grouped_e = df_joined.groupby(by="ID_e", as_index=False).apply(lambda g: g.sample(n=1)).set_index(keys=["ID_e", "ID_a"], drop=False).sort_index()
    df_joined_grouped_a = df_joined_grouped_a.rename(columns={"ID_e": "ID_CF", "Person_e": "Person_CF", "Sentence_e": "Sentence_CF",
                                                              "ID_a": "ID_F", "Person_a": "Person_F", "Sentence_a": "Sentence_F",
                                                              "Race_e": "Race_CF_label", "Race_a": "Race_F_label"})
    df_joined_grouped_e = df_joined_grouped_e.rename(columns={"ID_a": "ID_CF", "Person_a": "Person_CF", "Sentence_a": "Sentence_CF",
                                                              "ID_e": "ID_F", "Person_e": "Person_F", "Sentence_e": "Sentence_F",
                                                              "Race_e": "Race_F_label", "Race_a": "Race_CF_label"})
    df_final = pd.concat([df_joined_grouped_a, df_joined_grouped_e]).set_index(keys=["ID_F", "ID_CF"]).sort_index()
    return df_final, df_joined_grouped_a, df_joined_grouped_e


@timer
def create_all_gender_datasets():
    df_final, _, _ = create_gender_datasets()
    print(df_final)
    print_text_stats(df_final, "Sentence_F")
    split_data(df_final, POMS_GENDER_DATASETS_DIR, "gender")

    df_final, _, _ = create_gender_enriched_datasets()
    print(df_final)
    print_text_stats(df_final, "Sentence_F")
    split_data(df_final, POMS_GENDER_DATASETS_DIR, "gender_enriched")


@timer
def create_all_race_datasets():
    df_final, _, _ = create_race_datasets()
    print(df_final)
    print_text_stats(df_final, "Sentence_F")
    split_data(df_final, POMS_RACE_DATASETS_DIR, "race")

    df_final, _, _ = create_race_enriched_datasets()
    print(df_final)
    print_text_stats(df_final, "Sentence_F")
    split_data(df_final, POMS_RACE_DATASETS_DIR, "race_enriched")


def create_all_datasets():
    create_all_gender_datasets()
    create_race_enriched_datasets()


if __name__ == "__main__":
    create_all_race_datasets()
