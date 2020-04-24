from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from constants import POMS_GENDER_DATASETS_DIR, POMS_RAW_DATA_DIR, POMS_RACE_DATASETS_DIR, RANDOM_SEED
from datasets_utils import split_data, print_text_stats, bias_gentle, bias_aggressive
from Timer import timer
from glob import glob
from random import choice
from tqdm import tqdm
import pandas as pd

LABELS = {'None': 0, 'anger': 1, 'fear': 2, 'joy': 3, 'sadness': 4}
BIASED_LABEL = "joy"
BIASING_FACTOR = 0.1

test_emotion_words_dict = {
    "joy": [
        "blissful", "joyous", "delighted", "overjoyed", "gleeful", "thankful", "festive", "ecstatic", "satisfied",
        "cheerful", "sunny", "elated", "jubilant", "jovial", "lighthearted", "glorious", "innocent", "gratified",
        "euphoric", "world", "playful", "courageous", "energetic", "liberated", "optimistic", "frisky", "animated",
        "spirited", "thrilled", "intelligent", "exhilarated", "spunky", "youthful", "vigorous", "tickled", "creative",
        "constructive", "helpful", "resourceful", "comfortable", "pleased", "encouraged", "surprised", "content",
        "serene", "bright", "blessed", "vibrant", "bountiful", "glowing"
    ],
    "anger": [
        "Ordeal", "Outrageousness", "Provoke", "Repulsive", "Scandal", "Severe", "Shameful", "Shocking", "Terrible", "Tragic",
        "Unreliable", "Unstable", "Wicked", "Aggravate", "Agony", "Appalled", "Atrocious", "Corrupting", "Damaging",
        "Deplorable", "Disadvantages", "Disastrous", "Disgusted", "Dreadful", "Eliminate", "Harmful", "Harsh", "Inconsiderate",
        "enraged", "offensive", "aggressive", "frustrated", "controlling", "resentful", "malicious", "infuriated", "critical",
        "violent", "vindictive", "sadistic", "spiteful", "furious", "agitated", "antagonistic", "repulsed", "quarrelsome",
        "venomous", "rebellious", "exasperated", "impatient", "contrary", "condemning", "seething", "scornful", "sarcastic",
        "poisonous", "jealous", "revengeful", "retaliating", "reprimanding", "powerless", "despicable", "desperate", "alienated",
        "pessimistic", "dejected", "vilified", "unjustified", "violated"
    ],
    "sadness": [
        "bitter", "dismal", "heartbroken", "melancholy", "mournful", "pessimistic", "somber", "sorrowful", "sorry", "wistful",
        "bereaved", "blue", "cheerless", "dejected", "despairing", "despondent", "disconsolate", "distressed", "doleful",
        "down", "downcast", "forlorn", "glum", "grieved", "heartsick", "heavyhearted", "hurting", "languishing",
        "low", "lugubrious", "morbid", "morose", "pensive", "troubled", "weeping", "woebegone",
    ],
    "fear": [
        "angst", "anxiety", "concern", "despair", "dismay", "doubt", "dread", "horror", "jitters", "panic", "scare",
        "suspicion", "terror", "unease", "uneasiness", "worry", "abhorrence", "agitation", "aversion", "awe", "consternation",
        "cowardice", "creeps", "discomposure", "disquietude", "distress", "faintheartedness", "foreboding", "fright", "funk",
        "misgiving", "nightmare", "phobia", "presentiment", "qualm", "reverence", "revulsion", "timidity", "trembling",
        "tremor", "trepidation", "chickenheartedness", "recreancy"
    ],
    "nan": ["nan"]
}


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
    split_data(df_biased, output_dir, f"{dataset_type}_biased_{biased_label}_{biasing_method.__name__}", label_column)


@timer
def replace_test_set_emotion_words(output_dir: str):
    test_sets = glob(f"{output_dir}/*noisy*_test.csv")
    for dataset in tqdm(test_sets):
        df = pd.read_csv(dataset, header=0, encoding="utf-8").set_index(keys=["ID_F", "ID_CF"])
        df["new_Emotion_word"] = df["Emotion"].apply(lambda emotion:
                                                     str(choice(test_emotion_words_dict[str(emotion)])).lower())
        df["Sentence_F"] = df.apply(lambda row:
                                    str(row["Sentence_F"]).replace(str(row["Emotion_word"]),
                                                                   str(row["new_Emotion_word"])), axis=1)
        df["Sentence_CF"] = df.apply(lambda row:
                                     str(row["Sentence_CF"]).replace(str(row["Emotion_word"]),
                                                                     str(row["new_Emotion_word"])), axis=1)
        # Maybe we shouldn't replace the emotion word for CF to create a F vs CF bigger effect ?
        df.sort_index().to_csv(dataset)


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

        if corpus_type == "enriched_noisy":
            replace_test_set_emotion_words(output_dir)


@timer
def main():
    parser = ArgumentParser()
    parser.add_argument("--corpus_type", type=str, default='',
                        help="Corpus type can be: '', 'enriched', 'enriched_noisy', 'enriched_full'")
    args = parser.parse_args()
    create_all_datasets(args.corpus_type)


if __name__ == "__main__":
    main()
