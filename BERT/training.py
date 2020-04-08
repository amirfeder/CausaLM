from constants import SENTIMENT_RAW_DATA_DIR, SENTIMENT_EXPERIMENTS_DIR, POMS_EXPERIMENTS_DIR, POMS_GENDER_DATASETS_DIR, POMS_RACE_DATASETS_DIR
from pytorch_lightning import Trainer
from BERT.networks import LightningBertPretrainedClassifier, LightningHyperparameters
from BERT.predict import test_adj_models, print_final_metrics, test_genderace_models
from Timer import timer
from argparse import ArgumentParser
from typing import Dict
import torch

# LOGGER = init_logger("OOB_training")
DOMAIN = "movies"
MODE = "OOB_F"
BERT_STATE_DICT = None
TREATMENT = "adj"
TEXT_COLUMN = "review"
LABEL_COLUMN = "label"
DATASET_DIR = f"{SENTIMENT_RAW_DATA_DIR}/{DOMAIN}"
# DEVICE = get_free_gpu()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
### Constants
PAD_ID = 0
BATCH_SIZE = 256
ACCUMULATE = 4
DROPOUT = 0.1
EPOCHS = 50
FP16 = False

HYPERPARAMETERS = {
    "data_path": DATASET_DIR,
    "treatment": TREATMENT,
    "text_column": TEXT_COLUMN,
    "label_column": LABEL_COLUMN,
    "bert_params": {
        "batch_size": BATCH_SIZE,
        "dropout": DROPOUT,
        "bert_state_dict": BERT_STATE_DICT,
        "name": MODE
    }
}


def main():
    parser = ArgumentParser()
    parser.add_argument("--treatment", type=str, required=False, default="gender", help="Specify treatment for experiments: adj, gender, gender_enriched, race, race_enriched")
    args = parser.parse_args()
    if "gender" in args.treatment or "race" in args.treatment:
        train_all_genderace_models(args.treatment)
    elif "adj" in args.treatment:
        train_adj_models(args.treatment)


@timer
def bert_train_eval(hparams, output_dir):
    print(f"Training for {hparams['epochs']} epochs")
    trainer = Trainer(gpus=1 if DEVICE.type == "cuda" else 0,
                      default_save_path=output_dir,
                      show_progress_bar=True,
                      accumulate_grad_batches=hparams["accumulate"],
                      max_nb_epochs=hparams["epochs"],
                      early_stop_callback=None)
    hparams["output_path"] = trainer.logger.experiment.log_dir.rstrip('tf')
    model = LightningBertPretrainedClassifier(LightningHyperparameters(hparams))
    trainer.fit(model)
    trainer.test()
    print_final_metrics(hparams['bert_params']['name'], trainer.tqdm_metrics)
    return model


@timer
def train_adj_models():
    # Factual OOB BERT Model training
    OUTPUT_DIR = f"{SENTIMENT_EXPERIMENTS_DIR}/{TREATMENT}/{DOMAIN}/OOB_F"
    factual_oob_model = bert_train_eval(HYPERPARAMETERS, OUTPUT_DIR)
    # CounterFactual OOB BERT Model training
    OUTPUT_DIR = f"{SENTIMENT_EXPERIMENTS_DIR}/{TREATMENT}/{DOMAIN}/OOB_CF"
    HYPERPARAMETERS["text_column"] = "no_adj_review"
    HYPERPARAMETERS["bert_params"]["name"] = "OOB_CF"
    counterfactual_oob_model = bert_train_eval(HYPERPARAMETERS, OUTPUT_DIR)
    test_adj_models(factual_oob_model, counterfactual_oob_model)


@timer
def train_genderace_models(hparams: Dict):
    print(f"Training {hparams['treatment']} models")
    # Factual POMS BERT Model training
    hparams["bert_params"]["label_size"] = 5
    hparams["bert_params"]["name"] = "POMS_F"
    OUTPUT_DIR = f"{POMS_EXPERIMENTS_DIR}/{hparams['treatment']}/{hparams['bert_params']['name']}"
    hparams["text_column"] = "Sentence_F"
    factual_poms_model = bert_train_eval(hparams, OUTPUT_DIR)
    # CounterFactual POMS BERT Model training
    hparams["bert_params"]["name"] = "POMS_CF"
    OUTPUT_DIR = f"{POMS_EXPERIMENTS_DIR}/{hparams['treatment']}/{hparams['bert_params']['name']}"
    hparams["text_column"] = "Sentence_CF"
    counterfactual_poms_model = bert_train_eval(hparams, OUTPUT_DIR)
    # Factual CONTROL BERT Model training
    hparams["bert_params"]["label_size"] = 2
    hparams["bert_params"]["name"] = "CONTROL_F"
    OUTPUT_DIR = f"{POMS_EXPERIMENTS_DIR}/{hparams['treatment']}/{hparams['bert_params']['name']}"
    hparams["text_column"] = "Sentence_F"
    hparams["label_column"] = "Gender_F" if "gender" in hparams['treatment'] else "Race_F"
    factual_control_model = bert_train_eval(hparams, OUTPUT_DIR)
    # CounterFactual CONTROL BERT Model training
    hparams["bert_params"]["name"] = "CONTROL_CF"
    OUTPUT_DIR = f"{POMS_EXPERIMENTS_DIR}/{hparams['treatment']}/{hparams['bert_params']['name']}"
    hparams["text_column"] = "Sentence_CF"
    hparams["label_column"] = "Gender_CF" if "gender" in hparams['treatment'] else "Race_CF"
    counterfactual_control_model = bert_train_eval(hparams, OUTPUT_DIR)
    test_genderace_models(hparams["treatment"], factual_poms_model, counterfactual_poms_model, factual_control_model, counterfactual_control_model)


@timer
def train_all_genderace_models(treatment: str):
    HYPERPARAMETERS = {
        "data_path": POMS_GENDER_DATASETS_DIR if "gender" in treatment else POMS_RACE_DATASETS_DIR,
        "treatment": treatment,
        "text_column": "Sentence_F",
        "label_column": "label",
        "epochs": EPOCHS,
        "accumulate": ACCUMULATE,
        "bert_params": {
            "batch_size": BATCH_SIZE,
            "dropout": DROPOUT,
            "bert_state_dict": BERT_STATE_DICT,
            "label_size": 5,
            "name": "POMS_F"
        }
    }
    train_genderace_models(HYPERPARAMETERS)
    HYPERPARAMETERS["treatment"] = f"{treatment}_biased_joy_gentle"
    train_genderace_models(HYPERPARAMETERS)
    HYPERPARAMETERS["treatment"] = f"{treatment}_biased_joy_aggressive"
    train_genderace_models(HYPERPARAMETERS)


if __name__ == "__main__":
    main()
