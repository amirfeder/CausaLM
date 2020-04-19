from constants import POMS_GENDER_DATASETS_DIR, POMS_RAW_DATA_DIR, POMS_RACE_DATASETS_DIR
from datasets_utils import split_data, print_text_stats
from Timer import timer
import pandas as pd


# @timer
# def create_gender_datasets():
#     df = pd.read_csv(f"{POMS_RAW_DATA_DIR}/Equity-Evaluation-Corpus.csv", header=0,
#                      converters={"ID": lambda i: int(i.split("-")[-1])})
#     df = df.set_index(keys="ID", drop=False).sort_index()
#     emotion_labels = sorted([str(i) for i in df["Emotion"].unique() if str(i) != "nan"])
#     emotion_labels_dict = {label: i+1 for i, label in enumerate(emotion_labels)}
#     df_female = df[df["Gender"] == "female"].sort_index()
#     df_male = df[df["Gender"] == "male"].sort_index()
#     df_joined = pd.merge(df_female, df_male, left_on=["Template", "Race", "Emotion", "Emotion word"],
#                          right_on=["Template", "Race", "Emotion", "Emotion word"], how="inner",
#                          suffixes=("_1", "_0"), sort=True)
#     df_joined["POMS_label"] = df_joined["Emotion"].apply(lambda label: emotion_labels_dict.get(str(label), 0))
#     df_joined["Gender_0"] = 0
#     df_joined["Gender_1"] = 1
#     df_joined = df_joined.rename(columns={"Emotion word": "Emotion_word"})
#     df_joined["Race_label"] = df_joined["Race"].apply(lambda t: int(str(t) == "African-American"))
#     df_joined = df_joined[["ID_1", "ID_0", "Person_1", "Person_0", "Sentence_1", "Sentence_0", "Gender_1", "Gender_0", "Template", "Race", "Race_label", "Emotion_word", "Emotion", "POMS_label"]]
#     df_joined_grouped_f = df_joined.groupby(by="ID_1", as_index=False).apply(lambda g: g.sample(n=1)).set_index(keys=["ID_1", "ID_0"], drop=False).sort_index()
#     df_joined = df_joined[["ID_0", "ID_1", "Person_0", "Person_1", "Sentence_0", "Sentence_1", "Gender_0", "Gender_1", "Template", "Race", "Race_label", "Emotion_word", "Emotion", "POMS_label"]]
#     df_joined_grouped_m = df_joined.groupby(by="ID_0", as_index=False).apply(lambda g: g.sample(n=1)).set_index(keys=["ID_0", "ID_1"], drop=False).sort_index()
#     df_joined_grouped_f = df_joined_grouped_f.rename(columns={"ID_0": "ID_CF", "Person_0": "Person_CF", "Sentence_0": "Sentence_CF",
#                                                               "ID_1": "ID_F", "Person_1": "Person_F", "Sentence_1": "Sentence_F",
#                                                               "Gender_0": "Gender_CF_label", "Gender_1": "Gender_F_label"})
#     df_joined_grouped_m = df_joined_grouped_m.rename(columns={"ID_1": "ID_CF", "Person_1": "Person_CF", "Sentence_1": "Sentence_CF",
#                                                               "ID_0": "ID_F", "Person_0": "Person_F", "Sentence_0": "Sentence_F",
#                                                               "Gender_0": "Gender_F_label", "Gender_1": "Gender_CF_label"})
#     df_final = pd.concat([df_joined_grouped_f, df_joined_grouped_m]).set_index(keys=["ID_F", "ID_CF"]).sort_index()
#     return df_final, df_joined_grouped_f, df_joined_grouped_m
#
#
# @timer
# def create_gender_enriched_datasets():
#     df = pd.read_csv(f"{POMS_RAW_DATA_DIR}/Equity-Evaluation-Corpus_enriched.csv", header=0).set_index(keys="ID", drop=False).sort_index()
#     emotion_labels = sorted([str(i) for i in df["Emotion"].unique() if str(i) != "nan"])
#     emotion_labels_dict = {label: i+1 for i, label in enumerate(emotion_labels)}
#     df_female = df[df["Gender"] == "female"].sort_index()
#     df_male = df[df["Gender"] == "male"].sort_index()
#     df_joined = pd.merge(df_female, df_male, left_on=["Template", "Race", "Emotion", "Emotion_word"],
#                          right_on=["Template", "Race", "Emotion", "Emotion_word"], how="inner",
#                          suffixes=("_1", "_0"), sort=True)
#     df_joined["POMS_label"] = df_joined["Emotion"].apply(lambda label: emotion_labels_dict.get(str(label), 0))
#     df_joined["Gender_0"] = 0
#     df_joined["Gender_1"] = 1
#     df_joined["Race_label"] = df_joined["Race"].apply(lambda t: int(str(t) == "African-American"))
#     df_joined = df_joined[["ID_1", "ID_0", "Person_1", "Person_0", "Sentence_1", "Sentence_0", "Gender_1", "Gender_0", "Race", "Race_label", "Emotion_word", "Emotion", "POMS_label"]]
#     df_joined_grouped_f = df_joined.groupby(by="ID_1", as_index=False).apply(lambda g: g.sample(n=1)).set_index(keys=["ID_1", "ID_0"], drop=False).sort_index()
#     df_joined = df_joined[["ID_0", "ID_1", "Person_0", "Person_1", "Sentence_0", "Sentence_1", "Gender_0", "Gender_1", "Race", "Race_label", "Emotion_word", "Emotion", "POMS_label"]]
#     df_joined_grouped_m = df_joined.groupby(by="ID_0", as_index=False).apply(lambda g: g.sample(n=1)).set_index(keys=["ID_0", "ID_1"], drop=False).sort_index()
#     df_joined_grouped_f = df_joined_grouped_f.rename(columns={"ID_0": "ID_CF", "Person_0": "Person_CF", "Sentence_0": "Sentence_CF",
#                                                               "ID_1": "ID_F", "Person_1": "Person_F", "Sentence_1": "Sentence_F",
#                                                               "Gender_0": "Gender_CF_label", "Gender_1": "Gender_F_label"})
#     df_joined_grouped_m = df_joined_grouped_m.rename(columns={"ID_1": "ID_CF", "Person_1": "Person_CF", "Sentence_1": "Sentence_CF",
#                                                               "ID_0": "ID_F", "Person_0": "Person_F", "Sentence_0": "Sentence_F",
#                                                               "Gender_0": "Gender_F_label", "Gender_1": "Gender_CF_label"})
#     df_final = pd.concat([df_joined_grouped_f, df_joined_grouped_m]).set_index(keys=["ID_F", "ID_CF"]).sort_index()
#     return df_final, df_joined_grouped_f, df_joined_grouped_m
#
#
# @timer
# def create_race_datasets():
#     df = pd.read_csv(f"{POMS_RAW_DATA_DIR}/Equity-Evaluation-Corpus.csv", header=0,
#                      converters={"ID": lambda i: int(i.split("-")[-1])})
#     df = df.set_index(keys="ID", drop=False).sort_index()
#     emotion_labels = sorted([str(i) for i in df["Emotion"].unique() if str(i) != "nan"])
#     emotion_labels_dict = {label: i+1 for i, label in enumerate(emotion_labels)}
#     df_afro = df[df["Race"] == "African-American"].sort_index()
#     df_euro = df[df["Race"] == "European"].sort_index()
#     df_joined = pd.merge(df_afro, df_euro, left_on=["Template", "Gender", "Emotion", "Emotion word"],
#                          right_on=["Template", "Gender", "Emotion", "Emotion word"], how="inner",
#                          suffixes=("_1", "_0"), sort=True)
#     df_joined["POMS_label"] = df_joined["Emotion"].apply(lambda label: emotion_labels_dict.get(str(label), 0))
#     df_joined["Race_0"] = 0
#     df_joined["Race_1"] = 1
#     df_joined = df_joined.rename(columns={"Emotion word": "Emotion_word"})
#     df_joined["Gender_label"] = df_joined["Gender"].apply(lambda t: int(str(t) == "female"))
#     df_joined = df_joined[["ID_1", "ID_0", "Person_1", "Person_0", "Sentence_1", "Sentence_0", "Race_1", "Race_0", "Template", "Gender", "Gender_label", "Emotion_word", "Emotion", "POMS_label"]]
#     df_joined_grouped_a = df_joined.groupby(by="ID_1", as_index=False).apply(lambda g: g.sample(n=1)).set_index(keys=["ID_1", "ID_0"], drop=False).sort_index()
#     df_joined = df_joined[["ID_0", "ID_1", "Person_0", "Person_1", "Sentence_0", "Sentence_1", "Race_0", "Race_1", "Template", "Gender", "Gender_label", "Emotion_word", "Emotion", "POMS_label"]]
#     df_joined_grouped_e = df_joined.groupby(by="ID_0", as_index=False).apply(lambda g: g.sample(n=1)).set_index(keys=["ID_0", "ID_1"], drop=False).sort_index()
#     df_joined_grouped_a = df_joined_grouped_a.rename(columns={"ID_0": "ID_CF", "Person_0": "Person_CF", "Sentence_0": "Sentence_CF",
#                                                               "ID_1": "ID_F", "Person_1": "Person_F", "Sentence_1": "Sentence_F",
#                                                               "Race_0": "Race_CF_label", "Race_1": "Race_F_label"})
#     df_joined_grouped_e = df_joined_grouped_e.rename(columns={"ID_1": "ID_CF", "Person_1": "Person_CF", "Sentence_1": "Sentence_CF",
#                                                               "ID_0": "ID_F", "Person_0": "Person_F", "Sentence_0": "Sentence_F",
#                                                               "Race_0": "Race_F_label", "Race_1": "Race_CF_label"})
#     df_final = pd.concat([df_joined_grouped_a, df_joined_grouped_e]).set_index(keys=["ID_F", "ID_CF"]).sort_index()
#     return df_final, df_joined_grouped_a, df_joined_grouped_e
#
#
# @timer
# def create_race_enriched_datasets():
#     df = pd.read_csv(f"{POMS_RAW_DATA_DIR}/Equity-Evaluation-Corpus_enriched.csv", header=0).set_index(keys="ID", drop=False).sort_index()
#     emotion_labels = sorted([str(i) for i in df["Emotion"].unique() if str(i) != "nan"])
#     emotion_labels_dict = {label: i+1 for i, label in enumerate(emotion_labels)}
#     df_afro = df[df["Race"] == "African-American"].sort_index()
#     df_euro = df[df["Race"] == "European"].sort_index()
#     df_joined = pd.merge(df_afro, df_euro, left_on=["Template", "Gender", "Emotion", "Emotion_word"],
#                          right_on=["Template", "Gender", "Emotion", "Emotion_word"], how="inner",
#                          suffixes=("_1", "_0"), sort=True)
#     df_joined["POMS_label"] = df_joined["Emotion"].apply(lambda label: emotion_labels_dict.get(str(label), 0))
#     df_joined["Race_0"] = 0
#     df_joined["Race_1"] = 1
#     df_joined["Gender_label"] = df_joined["Gender"].apply(lambda t: int(str(t) == "female"))
#     df_joined = df_joined[["ID_1", "ID_0", "Person_1", "Person_0", "Sentence_1", "Sentence_0", "Race_1", "Race_0", "Template", "Gender", "Gender_label", "Emotion_word", "Emotion", "POMS_label"]]
#     df_joined_grouped_a = df_joined.groupby(by="ID_1", as_index=False).apply(lambda g: g.sample(n=1)).set_index(keys=["ID_1", "ID_0"], drop=False).sort_index()
#     df_joined = df_joined[["ID_0", "ID_1", "Person_0", "Person_1", "Sentence_0", "Sentence_1", "Race_0", "Race_1", "Template", "Gender", "Gender_label", "Emotion_word", "Emotion", "POMS_label"]]
#     df_joined_grouped_e = df_joined.groupby(by="ID_0", as_index=False).apply(lambda g: g.sample(n=1)).set_index(keys=["ID_0", "ID_1"], drop=False).sort_index()
#     df_joined_grouped_a = df_joined_grouped_a.rename(columns={"ID_0": "ID_CF", "Person_0": "Person_CF", "Sentence_0": "Sentence_CF",
#                                                               "ID_1": "ID_F", "Person_1": "Person_F", "Sentence_1": "Sentence_F",
#                                                               "Race_0": "Race_CF_label", "Race_1": "Race_F_label"})
#     df_joined_grouped_e = df_joined_grouped_e.rename(columns={"ID_1": "ID_CF", "Person_1": "Person_CF", "Sentence_1": "Sentence_CF",
#                                                               "ID_0": "ID_F", "Person_0": "Person_F", "Sentence_0": "Sentence_F",
#                                                               "Race_0": "Race_F_label", "Race_1": "Race_CF_label"})
#     df_final = pd.concat([df_joined_grouped_a, df_joined_grouped_e]).set_index(keys=["ID_F", "ID_CF"]).sort_index()
#     return df_final, df_joined_grouped_a, df_joined_grouped_e


@timer
def create_poms_dataset(treatment: str, treatment_vals: tuple, enriched=True):
    corpus_file = f"{POMS_RAW_DATA_DIR}/Equity-Evaluation-Corpus{'_enriched' if enriched else ''}.csv"
    treatment_column = treatment.capitalize()
    if treatment == "gender":
        control_column = "Race"
        control_val = "African-American"
    else:
        control_column = "Gender"
        control_val = "female"
    if enriched:
        df = pd.read_csv(corpus_file, header=0).set_index(keys="ID", drop=False).sort_index()
    else:
        df = pd.read_csv(corpus_file, header=0, converters={"ID": lambda i: int(i.split("-")[-1])}).rename(columns={"Emotion word": "Emotion_word"}).set_index(keys="ID", drop=False).sort_index()
    emotion_labels = sorted([str(i) for i in df["Emotion"].unique() if str(i) != "nan"])
    emotion_labels_dict = {label: i+1 for i, label in enumerate(emotion_labels)}
    df_one = df[df[treatment_column] == treatment_vals[0]].sort_index()
    df_zero = df[df[treatment_column] == treatment_vals[1]].sort_index()
    join_columns = ["Template", control_column, "Emotion", "Emotion_word"]
    df_joined = pd.merge(df_one, df_zero, left_on=join_columns, right_on=join_columns, how="inner", suffixes=("_1", "_0"), sort=True)
    df_joined["POMS_label"] = df_joined["Emotion"].apply(lambda label: emotion_labels_dict.get(str(label), 0))
    df_joined[f"{treatment_column}_0"] = 0
    df_joined[f"{treatment_column}_1"] = 1
    df_joined[f"{control_column}_label"] = df_joined[control_column].apply(lambda t: int(str(t) == control_val))
    df_joined = df_joined[["ID_1", "ID_0", "Person_1", "Person_0", "Sentence_1", "Sentence_0", f"{treatment_column}_1", f"{treatment_column}_0", "Template", control_column, f"{control_column}_label", "Emotion_word", "Emotion", "POMS_label"]]
    df_joined_grouped_one = df_joined.groupby(by="ID_1", as_index=False).apply(lambda g: g.sample(n=1)).set_index(keys=["ID_1", "ID_0"], drop=False).sort_index()
    df_joined = df_joined[["ID_0", "ID_1", "Person_0", "Person_1", "Sentence_0", "Sentence_1", f"{treatment_column}_0", f"{treatment_column}_1", "Template", control_column, f"{control_column}_label", "Emotion_word", "Emotion", "POMS_label"]]
    df_joined_grouped_zero = df_joined.groupby(by="ID_0", as_index=False).apply(lambda g: g.sample(n=1)).set_index(keys=["ID_0", "ID_1"], drop=False).sort_index()
    df_joined_grouped_one = df_joined_grouped_one.rename(columns={"ID_0": "ID_CF", "Person_0": "Person_CF", "Sentence_0": "Sentence_CF",
                                                              "ID_1": "ID_F", "Person_1": "Person_F", "Sentence_1": "Sentence_F",
                                                              f"{treatment_column}_0": f"{treatment_column}_CF_label",
                                                              f"{treatment_column}_1": f"{treatment_column}_F_label"})
    df_joined_grouped_zero = df_joined_grouped_zero.rename(columns={"ID_1": "ID_CF", "Person_1": "Person_CF", "Sentence_1": "Sentence_CF",
                                                              "ID_0": "ID_F", "Person_0": "Person_F", "Sentence_0": "Sentence_F",
                                                              f"{treatment_column}_0": f"{treatment_column}_F_label",
                                                              f"{treatment_column}_1": f"{treatment_column}_CF_label"})
    df_final = pd.concat([df_joined_grouped_one, df_joined_grouped_zero]).set_index(keys=["ID_F", "ID_CF"]).sort_index()
    return df_final, df_joined_grouped_one, df_joined_grouped_zero


# @timer
# def create_all_gender_datasets():
#     treatment = "gender"
#     treatment_vals = ("female", "male")
#     output_dir = POMS_GENDER_DATASETS_DIR
#     df_final, _, _ = create_poms_dataset(treatment, treatment_vals, False)
#     print(df_final)
#     print_text_stats(df_final, "Sentence_F")
#     split_data(df_final, output_dir, treatment)
#
#     df_final, _, _ = create_poms_dataset(treatment, treatment_vals, True)
#     print(df_final)
#     print_text_stats(df_final, "Sentence_F")
#     split_data(df_final, output_dir, f"{treatment}_enriched")
#
#
# @timer
# def create_all_race_datasets():
#     treatment = "race"
#     treatment_vals = ("African-American", "European")
#     output_dir = POMS_RACE_DATASETS_DIR
#     df_final, _, _ = create_poms_dataset(treatment, treatment_vals, False)
#     print(df_final)
#     print_text_stats(df_final, "Sentence_F")
#     split_data(df_final, output_dir, treatment)
#
#     df_final, _, _ = create_poms_dataset(treatment, treatment_vals, True)
#     print(df_final)
#     print_text_stats(df_final, "Sentence_F")
#     split_data(df_final, output_dir, f"{treatment}_enriched")


@timer
def create_all_datasets():
    for treatment, treatment_vals, output_dir in zip(("gender", "race"),
                                                     (("female", "male"), ("African-American", "European")),
                                                     (POMS_GENDER_DATASETS_DIR, POMS_RACE_DATASETS_DIR)):
        for enriched in (False, True):
            print(f"Creating {treatment.capitalize()}{' enriched' if enriched else ''} dataset")
            df_final, _, _ = create_poms_dataset(treatment, treatment_vals, enriched)
            print(df_final)
            print_text_stats(df_final, "Sentence_F")
            split_data(df_final, output_dir, f"{treatment}{'_enriched' if enriched else ''}", "POMS_label")


if __name__ == "__main__":
    create_all_datasets()
