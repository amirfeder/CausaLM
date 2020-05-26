from constants import POMS_EXPERIMENTS_DIR, POMS_GENDER_DATASETS_DIR, \
    POMS_RACE_DATASETS_DIR, MAX_POMS_SEQ_LENGTH, POMS_GENDER_DATA_DIR, POMS_RACE_DATA_DIR
from pytorch_lightning import Trainer
from BERT.bert_text_classifier import LightningBertPretrainedClassifier, LightningHyperparameters
from POMS_GendeRace.pipeline.predict import print_final_metrics, predict_genderace_models
from Timer import timer
from argparse import ArgumentParser
from typing import Dict
import torch

# LOGGER = init_logger("OOB_training")
from utils import init_logger

# DEVICE = get_free_gpu()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
### Constants
BATCH_SIZE = 200
ACCUMULATE = 4
DROPOUT = 0.1
EPOCHS = 50
FP16 = False


def main():
    parser = ArgumentParser()
    parser.add_argument("--treatment", type=str, required=True, default="gender",
                        help="Specify treatment for experiments: gender, race")
    parser.add_argument("--corpus_type", type=str, required=False, default="",
                        help="Corpus type can be: '', enriched, enriched_noisy, enriched_full")
    parser.add_argument("--group", type=str, required=True, default="F",
                        help="Specify data group for experiments: F (factual) or CF (counterfactual)")
    parser.add_argument("--pretrained_epoch", type=int, required=False, default=0,
                        help="Specify epoch for pretrained models: 0-4")
    args = parser.parse_args()
    if args.treatment in ("gender", "race"):
        train_all_genderace_models(args.treatment, args.corpus_type, args.group, args.pretrained_epoch)


@timer
def bert_train_eval(hparams, output_dir):
    trainer = Trainer(gpus=1 if DEVICE.type == "cuda" else 0,
                      default_save_path=output_dir,
                      show_progress_bar=True,
                      accumulate_grad_batches=hparams["accumulate"],
                      max_nb_epochs=hparams["epochs"],
                      early_stop_callback=None)
    hparams["output_path"] = trainer.logger.experiment.log_dir.rstrip('tf')
    logger = init_logger("training", hparams["output_path"])
    logger.info(f"Training for {hparams['epochs']} epochs")
    model = LightningBertPretrainedClassifier(LightningHyperparameters(hparams))
    trainer.fit(model)
    trainer.test()
    print_final_metrics(hparams['bert_params']['name'], trainer.tqdm_metrics, logger)
    return model


@timer
def train_genderace_models_unit(hparams: Dict, task, group):
    label_column = f"{task}_label"
    label_size = 2
    if task == "POMS":
        label_size = 5
    elif hparams["treatment"].startswith(task.lower()):
        label_column = f"{task}_{group}_label"
    hparams["label_column"] = label_column
    hparams["bert_params"]["label_size"] = label_size
    hparams["text_column"] = f"Sentence_{group}"
    if hparams["bert_params"]["bert_state_dict"]:
        hparams["bert_params"]["name"] = f"{task}_{group}_{hparams['treatment'].split('_')[0]}_treated"
    else:
        hparams["bert_params"]["name"] = f"{task}_{group}"
    OUTPUT_DIR = f"{POMS_EXPERIMENTS_DIR}/{hparams['treatment']}/{hparams['bert_params']['name']}"
    model = bert_train_eval(hparams, OUTPUT_DIR)
    return model


@timer
def train_genderace_models(hparams: Dict, group: str, pretrained_epoch: int):
    print(f"Training {hparams['treatment']} models")
    poms_model = train_genderace_models_unit(hparams, "POMS", group)
    gender_model = train_genderace_models_unit(hparams, "Gender", group)
    race_model = train_genderace_models_unit(hparams, "Race", group)
    if hparams["bert_params"]["bert_state_dict"]:
        group = f"{group}_{hparams['treatment'].split('_')[0]}_treated"
    predict_genderace_models(hparams['treatment'], group, pretrained_epoch,
                             poms_model, gender_model, race_model, hparams["bert_params"]["bert_state_dict"])


@timer
def train_all_genderace_models(treatment: str, corpus_type: str, group: str, pretrained_epoch: int):
    if corpus_type:
        treatment = f"{treatment}_{corpus_type}"
        state_dict_dir = f"model_{corpus_type}"
    else:
        state_dict_dir = "model"
    if pretrained_epoch is not None:
        state_dict_dir = f"{state_dict_dir}/epoch_{pretrained_epoch}"
    if treatment.startswith("gender"):
        data_path = POMS_GENDER_DATASETS_DIR
        pretrained_treated_model_dir = f"{POMS_GENDER_DATA_DIR}/{state_dict_dir}"
    else:
        data_path = POMS_RACE_DATASETS_DIR
        pretrained_treated_model_dir = f"{POMS_RACE_DATA_DIR}/{state_dict_dir}"

    hparams = {
        "data_path": data_path,
        "treatment": treatment,
        "text_column": f"Sentence_{group}",
        "label_column": "POMS_label",
        "epochs": EPOCHS,
        "accumulate": ACCUMULATE,
        "max_seq_len": MAX_POMS_SEQ_LENGTH,
        "bert_params": {
            "batch_size": BATCH_SIZE,
            "dropout": DROPOUT,
            "bert_state_dict": None,
            "label_size": 5,
            "name": f"POMS_{group}"
        }
    }
    train_genderace_models(hparams, group, pretrained_epoch)
    hparams["treatment"] = f"{treatment}_bias_gentle_3"
    train_genderace_models(hparams, group, pretrained_epoch)
    hparams["treatment"] = f"{treatment}_bias_aggressive_3"
    train_genderace_models(hparams, group, pretrained_epoch)

    hparams["bert_params"]["bert_state_dict"] = f"{pretrained_treated_model_dir}/pytorch_model.bin"
    hparams["treatment"] = treatment
    train_genderace_models(hparams, group, pretrained_epoch)
    hparams["treatment"] = f"{treatment}_bias_gentle_3"
    train_genderace_models(hparams, group, pretrained_epoch)
    hparams["treatment"] = f"{treatment}_bias_aggressive_3"
    train_genderace_models(hparams, group, pretrained_epoch)


if __name__ == "__main__":
    main()
