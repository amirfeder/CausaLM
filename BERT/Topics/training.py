import json

from constants import SENTIMENT_EXPERIMENTS_DIR, MAX_SENTIMENT_SEQ_LENGTH, SENTIMENT_DOMAINS, \
    SENTIMENT_TOPICS_DOMAIN_TREAT_CONTROL_MAP_FILE, SENTIMENT_TOPICS_DATASETS_DIR, SENTIMENT_TOPICS_PRETRAIN_ITX_DIR
from pytorch_lightning import Trainer
from BERT.bert_text_classifier import LightningBertPretrainedClassifier, LightningHyperparameters
from BERT.Topics.predict import print_final_metrics, predict_models
from Timer import timer
from argparse import ArgumentParser
from typing import Dict
import torch

# LOGGER = init_logger("OOB_training")
from utils import init_logger

# DEVICE = get_free_gpu()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
### Constants
BATCH_SIZE = 128
ACCUMULATE = 4
DROPOUT = 0.1
EPOCHS = 50
FP16 = False


def main():
    parser = ArgumentParser()
    parser.add_argument("--domain", type=str, default="books",
                        choices=("movies", "books", "dvd", "kitchen", "electronics", "all")),
    parser.add_argument("--group", type=str, required=False, default="F",
                        help="Specify data group for experiments: F (factual) or CF (counterfactual)")
    parser.add_argument("--pretrained_epoch", type=int, required=False, default=0,
                        help="Specify epoch for pretrained models: 0-4")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="Number of epochs to train for")
    args = parser.parse_args()

    if args.domain == "all":
        for domain in SENTIMENT_DOMAINS:
            train_all_models(args, domain)
    else:
        train_all_models(args, args.domain)


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
def train_models_unit(hparams: Dict, task, group):
    label_size = 2
    if task == "Sentiment":
        label_column = f"{task.lower()}_label"
    elif task == "ITT":
        label_column = hparams["treatment_column"]
    else:
        label_column = hparams["control_column"]
    hparams["label_column"] = label_column
    hparams["bert_params"]["label_size"] = label_size
    if hparams["bert_params"]["bert_state_dict"]:
        hparams["bert_params"]["name"] = f"{task}_{group}_topic_{hparams['treatment_column'].split('')[1]}_treated_topic_{hparams['control_column'].split('')[1]}_controlled"
    else:
        hparams["bert_params"]["name"] = f"{task}_{group}"
    OUTPUT_DIR = f"{SENTIMENT_EXPERIMENTS_DIR}/{hparams['treatment']}/{hparams['domain']}/{hparams['bert_params']['name']}"
    model = bert_train_eval(hparams, OUTPUT_DIR)
    return model


@timer
def train_models(hparams: Dict, group: str, pretrained_epoch: int):
    print(f"Training {hparams['treatment']} {hparams['domain']} models")
    sentiment_model = train_models_unit(hparams, "Sentiment", group)
    itt_model = train_models_unit(hparams, "ITT", group)
    itc_model = train_models_unit(hparams, "ITC", group)
    if hparams["bert_params"]["bert_state_dict"]:
        group = f"{group}_topic_{hparams['treatment_column'].split('')[1]}_treated_topic_{hparams['treatment_column'].split('')[1]}_controlled"
    predict_models(hparams['treatment'], hparams['domain'], group, pretrained_epoch,
                   sentiment_model, itt_model, itc_model, hparams["bert_params"]["bert_state_dict"])


@timer
def train_all_models(args, domain: str):

    with open(SENTIMENT_TOPICS_DOMAIN_TREAT_CONTROL_MAP_FILE, "r") as jsonfile:
        domain_topic_treat_dict = json.load(jsonfile)

    treatment_topic = domain_topic_treat_dict[domain]["treated_topic"]
    control_topic = domain_topic_treat_dict[domain]["control_topics"][0]

    pretrained_treated_model_dir = f"{SENTIMENT_TOPICS_PRETRAIN_ITX_DIR}/{domain}/model"
    if args.pretrained_epoch is not None:
        pretrained_treated_model_dir = f"{pretrained_treated_model_dir}/epoch_{args.pretrained_epoch}"

    treatment = "topics"

    hparams = {
        "treatment": treatment,
        "data_path": SENTIMENT_TOPICS_DATASETS_DIR,
        "domain": domain,
        "treatment_column": f"{treatment_topic}_bin",
        "control_column": f"{control_topic}_bin",
        "text_column": "review",
        "label_column": "sentiment_label",
        "epochs": args.epochs,
        "accumulate": ACCUMULATE,
        "max_seq_len": MAX_SENTIMENT_SEQ_LENGTH,
        "bert_params": {
            "batch_size": args.batch_size,
            "dropout": DROPOUT,
            "bert_state_dict": None,
            "label_size": 2,
            "name": f"Sentiment_{args.group}"
        }
    }
    train_models(hparams, args.group, args.pretrained_epoch)
    hparams["treatment"] = f"{treatment}_bias_gentle_{treatment_topic}_1"
    train_models(hparams, args.group, args.pretrained_epoch)
    hparams["treatment"] = f"{treatment}_bias_aggressive_{treatment_topic}_1"
    train_models(hparams, args.group, args.pretrained_epoch)

    hparams["bert_params"]["bert_state_dict"] = f"{pretrained_treated_model_dir}/pytorch_model.bin"
    hparams["treatment"] = treatment
    train_models(hparams, args.group, args.pretrained_epoch)
    hparams["treatment"] = f"{treatment}_bias_gentle_{treatment_topic}_1"
    train_models(hparams, args.group, args.pretrained_epoch)
    hparams["treatment"] = f"{treatment}_bias_aggressive_{treatment_topic}_1"
    train_models(hparams, args.group, args.pretrained_epoch)


if __name__ == "__main__":
    main()
