import json
from argparse import ArgumentParser
from typing import Dict
from constants import SENTIMENT_TOPICS_DATASETS_DIR, SENTIMENT_EXPERIMENTS_DIR, \
    SENTIMENT_TOPICS_PRETRAIN_MLM_DIR, SENTIMENT_TOPICS_PRETRAIN_ITX_DIR, SENTIMENT_DOMAINS, \
    SENTIMENT_TOPICS_DOMAIN_TREAT_CONTROL_MAP_FILE
from pytorch_lightning import Trainer, LightningModule
from BERT.bert_text_classifier import LightningBertPretrainedClassifier, BertPretrainedClassifier
from os import listdir, path
from glob import glob
from copy import deepcopy
from Timer import timer
from utils import GoogleDriveHandler, send_email, init_logger
import torch


def get_checkpoint_file(ckpt_dir: str):
    for file in sorted(listdir(ckpt_dir)):
        if file.endswith(".ckpt"):
            return f"{ckpt_dir}/{file}"
    else:
        return None


def find_latest_model_checkpoint(models_dir: str):
    model_ckpt = None
    while not model_ckpt:
        model_versions = sorted(glob(models_dir), key=path.getctime)
        if model_versions:
            latest_model = model_versions.pop()
            model_ckpt_dir = f"{latest_model}/checkpoints"
            model_ckpt = get_checkpoint_file(model_ckpt_dir)
        else:
            raise FileNotFoundError(f"Couldn't find a model checkpoint in {models_dir}")
    return model_ckpt


def print_final_metrics(name: str, metrics: Dict, logger=None):
    if logger:
        logger.info(f"{name} Metrics:")
        for metric, val in metrics.items():
            logger.info(f"{metric}: {val:.4f}")
        logger.info("\n")
    else:
        print(f"{name} Metrics:")
        for metric, val in metrics.items():
            print(f"{metric}: {val:.4f}")
        print()


def main():
    parser = ArgumentParser()
    parser.add_argument("--domain", type=str, default="books",
                        choices=("movies", "books", "dvd", "kitchen", "electronics", "all")),
    parser.add_argument("--trained_group", type=str, required=False, default="F",
                        help="Specify data group for trained_models: F (factual) or CF (counterfactual)")
    parser.add_argument("--pretrained_epoch", type=int, required=False, default=0,
                        help="Specify epoch for pretrained models: 0-4")
    args = parser.parse_args()

    if args.domain == "all":
        for domain in SENTIMENT_DOMAINS:
            predict_all_models(args, domain)
    else:
        predict_all_models(args, args.domain)


@timer
def bert_treatment_test(model_ckpt, hparams, trainer, logger=None):
    if isinstance(model_ckpt, LightningModule):
        model = LightningBertPretrainedClassifier(deepcopy(model_ckpt.hparams))
        model.bert_classifier = deepcopy(model_ckpt.bert_classifier)
    else:
        logger.info(f"Loading model for {hparams['treatment']} {hparams['bert_params']['name']} from: {model_ckpt}")
        model = LightningBertPretrainedClassifier.load_from_checkpoint(model_ckpt)

    if hparams["bert_params"]["bert_state_dict"]:
        model.hparams.bert_params["bert_state_dict"] = hparams["bert_params"]["bert_state_dict"]
        model.bert_classifier.bert_state_dict = hparams["bert_params"]["bert_state_dict"]
        logger.info(f"Loading pretrained BERT model for {hparams['bert_params']['name']} from: {model.bert_classifier.bert_state_dict}")
        model.bert_classifier.bert = BertPretrainedClassifier.load_frozen_bert(model.bert_classifier.bert_pretrained_model,
                                                                               model.bert_classifier.bert_state_dict)

    # Update model hyperparameters
    model.hparams.output_path = hparams["output_path"]
    model.hparams.label_column = hparams["label_column"]
    model.hparams.bert_params["name"] = hparams['bert_params']['name']
    model.bert_classifier.name = hparams['bert_params']['name']

    model.freeze()
    trainer.test(model)
    print_final_metrics(hparams['bert_params']['name'], trainer.tqdm_metrics, logger)


@timer
def predict_models_unit(task, treatment, domain, trained_group, group, model_ckpt,
                        hparams, trainer, logger, pretrained_epoch, bert_state_dict):

    state_dict_dir = f"{domain}/model"
    if pretrained_epoch is not None:
        state_dict_dir = f"{state_dict_dir}/epoch_{pretrained_epoch}"

    # TODO: Finalize what CF is for topics and how to implement here

    label_size = 2
    if task == "Sentiment":
        label_column = f"{task.lower()}_label"
    elif "ITT" in task:
        label_column = hparams['treatment_column']
        task = "ITT"
    else:
        label_column = hparams['control_column']
        task = "ITC"

    hparams["label_column"] = label_column
    logger.info(f"Treatment: {treatment}")
    logger.info(f"Text Column: {hparams['text_column']}")
    logger.info(f"Label Column: {label_column}")
    logger.info(f"Label Size: {label_size}")
    hparams["bert_params"]["name"] = f"{task}_{group}_trained_{trained_group}"
    hparams["bert_params"]["bert_state_dict"] = bert_state_dict

    if not model_ckpt:
        model_name = f"{task}_{hparams['trained_group']}"
        models_dir = f"{SENTIMENT_EXPERIMENTS_DIR}/{hparams['treatment']}/{hparams['domain']}/{model_name}/lightning_logs/*"
        model_ckpt = find_latest_model_checkpoint(models_dir)

    # Group Task BERT Model training
    logger.info(f"Model: {hparams['bert_params']['name']}")
    bert_treatment_test(model_ckpt, hparams, trainer, logger)

    # Group Task BERT Model test with MLM LM
    hparams["bert_params"]["name"] = f"{task}_MLM_{group}_trained_{trained_group}"
    hparams["bert_params"]["bert_state_dict"] = f"{SENTIMENT_TOPICS_PRETRAIN_MLM_DIR}/{state_dict_dir}/pytorch_model.bin"
    logger.info(f"MLM Pretrained Model: {hparams['bert_params']['bert_state_dict']}")
    bert_treatment_test(model_ckpt, hparams, trainer, logger)

    if not bert_state_dict:
        # Group Task BERT Model test with Gender/Race treated LM
        hparams["bert_params"]["name"] = f"{task}_topic_{hparams['treatment_column'].split('_')[1]}_treated_topic_{hparams['control_column'].split('_')[1]}_controlled_{group}_trained_{trained_group}"
        hparams["bert_params"]["bert_state_dict"] = f"{SENTIMENT_TOPICS_PRETRAIN_ITX_DIR}/{state_dict_dir}/pytorch_model.bin"
        logger.info(f"Treated Pretrained Model: {hparams['bert_params']['bert_state_dict']}")
        bert_treatment_test(model_ckpt, hparams, trainer, logger)


@timer
def predict_models(treatment="topics", domain="books", trained_group="F", pretrained_epoch=None,
                   sentiment_model_ckpt=None, itt_model_ckpt=None, itc_model_ckpt=None,
                   bert_state_dict=None):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(SENTIMENT_TOPICS_DOMAIN_TREAT_CONTROL_MAP_FILE, "r") as jsonfile:
        domain_topic_treat_dict = json.load(jsonfile)
    treatment_topic = domain_topic_treat_dict[domain]["treated_topic"]
    control_topic = domain_topic_treat_dict[domain]["control_topics"][-1]
    hparams = {
        "treatment": treatment,
        "domain": domain,
        "trained_group": trained_group,
        "treatment_column": f"{treatment_topic}_bin",
        "control_column": f"{control_topic}_bin",
        "text_column": "review",
        "data_path": SENTIMENT_TOPICS_DATASETS_DIR,
        "bert_params": {
            "bert_state_dict": bert_state_dict
        }
    }
    OUTPUT_DIR = f"{SENTIMENT_EXPERIMENTS_DIR}/{treatment}/{domain}/COMPARE"
    trainer = Trainer(gpus=1 if DEVICE.type == "cuda" else 0,
                      default_save_path=OUTPUT_DIR,
                      show_progress_bar=True,
                      early_stop_callback=None)
    hparams["output_path"] = trainer.logger.experiment.log_dir.rstrip('tf')
    logger = init_logger(f"testing", hparams["output_path"])
    for task, model in zip(("Sentiment", "CONTROL_ITT", "CONTROL_ITC"),
                           (sentiment_model_ckpt, itt_model_ckpt, itc_model_ckpt)):
        for group in ("F",): #TODO: Finalize what CF is for topics
            predict_models_unit(task, treatment, domain, trained_group, group, model,
                                hparams, trainer, logger, pretrained_epoch, bert_state_dict)
    handler = GoogleDriveHandler()
    push_message = handler.push_files(hparams["output_path"])
    logger.info(push_message)
    send_email(push_message, treatment)


@timer
def predict_all_models(args, domain: str):

    treatment = "topics"

    with open(SENTIMENT_TOPICS_DOMAIN_TREAT_CONTROL_MAP_FILE, "r") as jsonfile:
        domain_topic_treat_dict = json.load(jsonfile)

    treatment_topic = domain_topic_treat_dict[domain]["treated_topic"]
    control_topic = domain_topic_treat_dict[domain]["control_topics"][-1]

    predict_models(treatment, domain, args.trained_group, args.pretrained_epoch)
    predict_models(f"{treatment}_bias_gentle_{treatment_topic}_1", domain, args.trained_group, args.pretrained_epoch)
    predict_models(f"{treatment}_bias_aggressive_{treatment_topic}_1", domain, args.trained_group, args.pretrained_epoch)

    pretrained_treated_model_dir = f"{SENTIMENT_TOPICS_PRETRAIN_ITX_DIR}/{domain}/model"
    if args.pretrained_epoch is not None:
        pretrained_treated_model_dir = f"{pretrained_treated_model_dir}/epoch_{args.pretrained_epoch}"

    bert_state_dict = f"{pretrained_treated_model_dir}/pytorch_model.bin"
    trained_group = f"{args.trained_group}_{treatment_topic}_treated_{control_topic}_controlled"
    predict_models(treatment, domain, trained_group, args.pretrained_epoch, bert_state_dict=bert_state_dict)
    predict_models(f"{treatment}_bias_gentle_{treatment_topic}_1", domain, trained_group, args.pretrained_epoch, bert_state_dict=bert_state_dict)
    predict_models(f"{treatment}_bias_aggressive_{treatment_topic}_1", domain, trained_group, args.pretrained_epoch, bert_state_dict=bert_state_dict)


if __name__ == "__main__":
    main()
