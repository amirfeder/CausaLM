from constants import SENTIMENT_EXPERIMENTS_DIR, MAX_SENTIMENT_SEQ_LENGTH, SENTIMENT_IMA_PRETRAIN_DATA_DIR, \
    SENTIMENT_RAW_DATA_DIR, ALL_SENTIMENT_DOMAINS
from pytorch_lightning import Trainer
from BERT.bert_text_classifier import LightningBertPretrainedClassifier, LightningHyperparameters
from BERT.bert_pos_tagger import LightningBertPOSTagger
from Sentiment_Adjectives.pipeline.predict import print_final_metrics, predict_models

from argparse import ArgumentParser
from typing import Dict
import torch

from datasets.utils import NUM_POS_TAGS_LABELS
from utils import init_logger

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Constants
BATCH_SIZE = 128
ACCUMULATE = 4
DROPOUT = 0.1
EPOCHS = 50
FP16 = False


def main():
    parser = ArgumentParser()
    parser.add_argument("--treatment", type=str, default="adj", choices=("adj",),
                        help="Specify treatment for experiments: adj")
    parser.add_argument("--domain", type=str, default="books", choices=("unified", "movies", "books", "dvd", "kitchen", "electronics", "all"),
                        help="Dataset Domain: unified, movies, books, dvd, kitchen, electronics")
    parser.add_argument("--group", type=str, default="F", choices=("F", "CF"),
                        help="Specify data group for experiments: F (factual) or CF (counterfactual)")
    parser.add_argument("--masking_method", type=str, default="double_num_adj", choices=("double_num_adj", "mlm_prob"),
                        help="Method of determining num masked tokens in sentence: mlm_prob or double_num_adj")
    parser.add_argument("--pretrained_epoch", type=int, default=0,
                        help="Specify epoch for pretrained models: 0-4")
    parser.add_argument("--pretrained_control", action="store_true",
                        help="Use pretraining model with control task")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="Number of epochs to train for")
    args = parser.parse_args()

    if args.domain == "all":
        for domain in ALL_SENTIMENT_DOMAINS:
            train_all_models(args, domain)
    else:
        train_all_models(args, args.domain)



def bert_train_eval(hparams, task, output_dir):
    trainer = Trainer(gpus=1 if DEVICE.type == "cuda" else 0,
                      default_save_path=output_dir,
                      show_progress_bar=True,
                      accumulate_grad_batches=hparams["accumulate"],
                      max_nb_epochs=hparams["epochs"],
                      early_stop_callback=None)
    hparams["output_path"] = trainer.logger.experiment.log_dir.rstrip('tf')
    logger = init_logger("training", hparams["output_path"])
    logger.info(f"Training {task} for {hparams['epochs']} epochs")
    if task == "Sentiment":
        hparams["bert_params"]["batch_size"] = hparams["batch_size"]
        model = LightningBertPretrainedClassifier(LightningHyperparameters(hparams))
    else:
        model = LightningBertPOSTagger(LightningHyperparameters(hparams))
    trainer.fit(model)
    trainer.test()
    print_final_metrics(hparams['bert_params']['name'], trainer.tqdm_metrics, logger)
    return model



def train_models_unit(hparams: Dict, task, group, pretrained_control):
    label_size = 2
    if task == "POS_Tagging":
        label_size = NUM_POS_TAGS_LABELS
        label_column = f"{task.lower()}_{group.lower()}_labels"
    elif task == "IMA":
        label_column = f"{task.lower()}_{group.lower()}_labels"
    else:
        label_column = f"{task.lower()}_label"

    hparams["label_column"] = label_column
    hparams["num_labels"] = label_size
    hparams["bert_params"]["label_size"] = label_size

    if hparams["bert_params"]["bert_state_dict"]:
        if pretrained_control:
            hparams["bert_params"]["name"] = f"{task}_{group}_ima_control_treated"
        else:
            hparams["bert_params"]["name"] = f"{task}_{group}_ima_treated"
    else:
        hparams["bert_params"]["name"] = f"{task}_{group}"

    OUTPUT_DIR = f"{SENTIMENT_EXPERIMENTS_DIR}/{hparams['treatment']}/{hparams['domain']}/{hparams['bert_params']['name']}"
    model = bert_train_eval(hparams, task, OUTPUT_DIR)
    return model



def train_models(hparams: Dict, group: str, pretrained_masking_method, pretrained_epoch: int, pretrained_control: bool):
    print(f"Training {hparams['treatment']} {hparams['domain']} models")
    sentiment_model = train_models_unit(hparams, "Sentiment", group, pretrained_control)
    ima_model = train_models_unit(hparams, "IMA", group, pretrained_control)
    pos_tagging_model = train_models_unit(hparams, "POS_Tagging", group, pretrained_control)
    if hparams["bert_params"]["bert_state_dict"]:
        if pretrained_control:
            group = f"{group}_ima_control_treated"
        else:
            group = f"{group}_ima_treated"
    predict_models(hparams['treatment'], hparams['domain'], group,
                   pretrained_masking_method, pretrained_epoch, pretrained_control,
                   sentiment_model, ima_model, pos_tagging_model,
                   hparams["bert_params"]["bert_state_dict"])



def train_all_models(args, domain: str):

    if args.group == "F":
        text_column = "review"
    else:
        text_column = "no_adj_review"

    if args.pretrained_control:
        pretrained_treated_model_dir = f"{SENTIMENT_IMA_PRETRAIN_DATA_DIR}/{args.masking_method}/{domain}/model_control"
    else:
        pretrained_treated_model_dir = f"{SENTIMENT_IMA_PRETRAIN_DATA_DIR}/{args.masking_method}/{domain}/model"

    if args.pretrained_epoch is not None:
        pretrained_treated_model_dir = f"{pretrained_treated_model_dir}/epoch_{args.pretrained_epoch}"

    data_path = f"{SENTIMENT_RAW_DATA_DIR}/{domain}"

    hparams = {
        "data_path": data_path,
        "treatment": args.treatment,
        "domain": domain,
        "masking_method": args.masking_method,
        "pretrain_conrol": args.pretrained_control,
        "text_column": text_column,
        "label_column": "sentiment_label",
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "accumulate": ACCUMULATE,
        "max_seq_len": MAX_SENTIMENT_SEQ_LENGTH,
        "num_labels": 2,
        "name": f"Sentiment_{args.group}",
        "bert_params": {
            "dropout": DROPOUT,
            "bert_state_dict": None,
            "label_size": 2,
            "name": f"Sentiment_{args.group}"
        }
    }
    train_models(hparams, args.group, args.masking_method, args.pretrained_epoch, args.pretrained_control)
    hparams["treatment"] = f"{args.treatment}_bias_gentle_ratio_adj_1"
    train_models(hparams, args.group, args.masking_method, args.pretrained_epoch, args.pretrained_control)
    hparams["treatment"] = f"{args.treatment}_bias_aggressive_ratio_adj_1"
    train_models(hparams, args.group, args.masking_method, args.pretrained_epoch, args.pretrained_control)

    hparams["bert_params"]["bert_state_dict"] = f"{pretrained_treated_model_dir}/pytorch_model.bin"
    hparams["treatment"] = args.treatment
    train_models(hparams, args.group, args.masking_method, args.pretrained_epoch, args.pretrained_control)
    hparams["treatment"] = f"{args.treatment}_bias_gentle_ratio_adj_1"
    train_models(hparams, args.group, args.masking_method, args.pretrained_epoch, args.pretrained_control)
    hparams["treatment"] = f"{args.treatment}_bias_aggressive_ratio_adj_1"
    train_models(hparams, args.group, args.masking_method, args.pretrained_epoch, args.pretrained_control)


if __name__ == "__main__":
    main()
