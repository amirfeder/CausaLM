from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from constants import POMS_GENDER_DATASETS_DIR, POMS_RAW_DATA_DIR, POMS_RACE_DATASETS_DIR, RANDOM_SEED
from datasets_utils import split_data, print_text_stats, bias_gentle, bias_aggressive
from Timer import timer
import pandas as pd

LABELS = {'None': 0, 'anger': 1, 'fear': 2, 'joy': 3, 'sadness': 4}
BIASED_LABEL = "joy"
BIASING_FACTOR = 0.1


@timer
def create_poms_dataset(treatment: str, treatment_vals: tuple, corpus_type: str = ''):
    if corpus_type:
        corpus_file = f"{POMS_RAW_DATA_DIR}/Equity-Evaluation-Corpus_{corpus_type}.csv"
    else:
        corpus_file = f"{POMS_RAW_DATA_DIR}/Equity-Evaluation-Corpus.csv"
    treatment_column = treatment.capitalize()
    if treatment == "gender":
        control_column = "Race"
        control_val = "African-American"
    else:
        control_column = "Gender"
        control_val = "female"
    if "enriched" in corpus_type:
        df = pd.read_csv(corpus_file, header=0, encoding='utf-8').set_index(keys="ID", drop=False).sort_index()
    else:
        df = pd.read_csv(corpus_file, header=0, encoding='utf-8', converters={"ID": lambda i: int(i.split("-")[-1])}).rename(columns={"Emotion word": "Emotion_word"}).set_index(keys="ID", drop=False).sort_index()
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


@timer
def create_biased_datasets(df_a, df_b, label_column, biased_label, biasing_factor, biasing_method, dataset_type="gender", output_dir=POMS_GENDER_DATASETS_DIR):
    df_biased = biasing_method(df_a, df_b, label_column, biased_label, biasing_factor)
    df_biased = df_biased.set_index(keys=["ID_F", "ID_CF"]).sort_index()
    print(df_biased)
    print_text_stats(df_biased, "Sentence_F")
    split_data(df_biased, output_dir, f"{dataset_type}_{biasing_method.__name__}_{biased_label}", label_column)


@timer
def create_all_datasets(corpus_type: str):
    label_column = "POMS_label"
    for treatment, treatment_vals, output_dir in zip(("gender", "race"),
                                                     (("female", "male"), ("African-American", "European")),
                                                     (POMS_GENDER_DATASETS_DIR, POMS_RACE_DATASETS_DIR)):

        print(f"Creating {treatment.capitalize()}{f' {corpus_type}' if corpus_type else ''} datasets")
        df_final, df_one, df_zero = create_poms_dataset(treatment, treatment_vals, corpus_type)
        print(df_final)
        print_text_stats(df_final, "Sentence_F")
        if corpus_type == "enriched_full":
            _, df_final = train_test_split(df_final, test_size=0.1, random_state=RANDOM_SEED, stratify=df_final[label_column])
            _, df_one = train_test_split(df_one, test_size=0.1, random_state=RANDOM_SEED, stratify=df_final[label_column])
            _, df_zero = train_test_split(df_zero, test_size=0.1, random_state=RANDOM_SEED, stratify=df_final[label_column])
        split_data(df_final, output_dir, f"{treatment}{f'_{corpus_type}' if corpus_type else ''}", label_column)

        print(f"Biasing {treatment.capitalize()}{f' {corpus_type}' if corpus_type else ''} dataset")
        for bias_method in (bias_aggressive, bias_gentle):
            create_biased_datasets(df_one, df_zero, label_column, LABELS[BIASED_LABEL], BIASING_FACTOR, bias_method,
                                   f"{treatment}{f'_{corpus_type}' if corpus_type else ''}", output_dir)


@timer
def main():
    parser = ArgumentParser()
    parser.add_argument("--corpus_type", type=str, default='',
                        help="Corpus type can be: '', 'enriched', 'enriched_noisy', 'enriched_full'")
    args = parser.parse_args()
    create_all_datasets(args.corpus_type)


if __name__ == "__main__":
    main()
