from argparse import ArgumentParser
from copy import deepcopy

from BERT.bert_text_classifier import LightningBertPretrainedClassifier, BertPretrainedClassifier
from BERT.bert_pos_tagger import LightningBertPOSTagger
from constants import SENTIMENT_EXPERIMENTS_DIR, MAX_SENTIMENT_SEQ_LENGTH, SENTIMENT_ADJECTIVES_PRETRAIN_IMA_DIR, \
    SENTIMENT_ADJECTIVES_DATASETS_DIR, SENTIMENT_ADJECTIVES_PRETRAIN_MLM_DIR
from pytorch_lightning import Trainer, LightningModule

from datasets.utils import NUM_POS_TAGS_LABELS
from utils import GoogleDriveHandler,  init_logger, print_final_metrics, find_latest_model_checkpoint
import torch


def main():
    parser = ArgumentParser()
    parser.add_argument("--treatment", type=str, default="adjectives", choices=("adjectives",),
                        help="Specify treatment for experiments")
    parser.add_argument("--group", type=str, default="F", choices=("F", "CF"),
                        help="Specify data group for experiments: F (factual) or CF (counterfactual)")
    parser.add_argument("--masking_method", type=str, default="double_num_adj", choices=("double_num_adj", "mlm_prob"),
                        help="Method of determining num masked tokens in sentence")
    parser.add_argument("--pretrained_epoch", type=int, default=0,
                        help="Specify epoch for pretrained models: 0-4")
    parser.add_argument("--pretrained_control", action="store_true",
                        help="Use pretraining model with control task")
    args = parser.parse_args()

    predict_all_models(args.treatment, args.group, args.masking_method, args.pretrained_epoch, args.pretrained_control)



def bert_treatment_test(model_ckpt, hparams, trainer, logger=None, task="Sentiment"):
    if task == "Sentiment":
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
        model.bert_classifier.name = hparams['bert_params']['name']
        model.bert_classifier.label_size = hparams["bert_params"]["label_size"]

    else:
        if isinstance(model_ckpt, LightningModule):
            model = LightningBertPOSTagger(deepcopy(model_ckpt.hparams))
            model.bert_token_classifier = deepcopy(model_ckpt.bert_token_classifier)
        else:
            logger.info(f"Loading model for {hparams['treatment']} {hparams['bert_params']['name']} from: {model_ckpt}")
            model = LightningBertPOSTagger.load_from_checkpoint(model_ckpt)

        if hparams["bert_params"]["bert_state_dict"]:
            model.hparams.num_labels = hparams["num_labels"]
            model.num_labels = hparams["num_labels"]
            model.hparams.bert_params["bert_state_dict"] = hparams["bert_params"]["bert_state_dict"]
            model.bert_state_dict = hparams["bert_params"]["bert_state_dict"]
            logger.info(f"Loading pretrained BERT model for {hparams['bert_params']['name']} from: {model.bert_state_dict}")
            model.bert_token_classifier.bert = LightningBertPOSTagger.load_pretrained_state_dict(model.bert_pretrained_model,
                                                                                                 model.bert_state_dict)

    # Update model hyperparameters
    model.hparams.max_seq_len = hparams["max_seq_len"]
    model.hparams.output_path = hparams["output_path"]
    model.hparams.label_column = hparams["label_column"]
    model.hparams.name = hparams['bert_params']['name']
    model.hparams.text_column = hparams["text_column"]
    model.hparams.bert_params["name"] = hparams['bert_params']['name']
    model.hparams.bert_params["label_size"] = hparams["bert_params"]["label_size"]

    model.freeze()
    trainer.test(model)
    print_final_metrics(hparams['bert_params']['name'], trainer.tqdm_metrics, logger)


def predict_models_unit(task, trained_group, group, model_ckpt, hparams, trainer, logger,
                        pretrained_masking_method, pretrained_epoch, pretrained_control, bert_state_dict):

    if group == "F":
        text_column = "review"
    else:
        text_column = "no_adj_review"

    label_size = 2
    if "POS_Tagging" in task:
        label_size = NUM_POS_TAGS_LABELS
        label_column = f"pos_tagging_{group.lower()}_labels"
        trained_task = "POS_Tagging"
    elif "IMA" in task:
        label_column = f"ima_{group.lower()}_labels"
        trained_task = "IMA"
    else:
        label_column = f"sentiment_label"
        trained_task = "Sentiment"

    hparams["max_seq_len"] = MAX_SENTIMENT_SEQ_LENGTH
    hparams["label_column"] = label_column
    hparams["num_labels"] = label_size
    hparams["bert_params"]["label_size"] = label_size
    hparams["text_column"] = text_column
    hparams["trained_group"] = trained_group
    logger.info(f"Treatment: {hparams['treatment']}")
    logger.info(f"Text Column: {hparams['text_column']}")
    logger.info(f"Label Column: {label_column}")
    logger.info(f"Label Size: {label_size}")
    hparams["bert_params"]["name"] = f"{task}_{group}_trained_{trained_group}"
    hparams["bert_params"]["bert_state_dict"] = bert_state_dict

    if not model_ckpt:
        model_name = f"{trained_task}_{trained_group}"
        models_dir = f"{SENTIMENT_EXPERIMENTS_DIR}/{hparams['treatment']}/{model_name}/lightning_logs/*"
        model_ckpt = find_latest_model_checkpoint(models_dir)

    # Group Task BERT Model training
    logger.info(f"Model: {hparams['bert_params']['name']}")
    bert_treatment_test(model_ckpt, hparams, trainer, logger, task)

    # Group Task BERT Model test with MLM LM
    hparams["bert_params"]["name"] = f"{task}_MLM_{group}_trained_{trained_group}"
    hparams["bert_params"]["bert_state_dict"] = f"{SENTIMENT_ADJECTIVES_PRETRAIN_MLM_DIR}/{pretrained_masking_method}/model/epoch_{pretrained_epoch}/pytorch_model.bin"
    logger.info(f"MLM Pretrained Model: {hparams['bert_params']['bert_state_dict']}")
    bert_treatment_test(model_ckpt, hparams, trainer, logger, task)

    if not bert_state_dict:

        if pretrained_control:
            treatment_method = "ima_control"
            state_dict_dir = f"{pretrained_masking_method}/model_control"
        else:
            treatment_method = "ima"
            state_dict_dir = f"{pretrained_masking_method}/model"

        if pretrained_epoch is not None:
            state_dict_dir = f"{state_dict_dir}/epoch_{pretrained_epoch}"

        pretrained_treated_model_dir = f"{SENTIMENT_ADJECTIVES_PRETRAIN_IMA_DIR}/{state_dict_dir}"
        # Group Task BERT Model test with Gender/Race treated LM
        hparams["bert_params"]["name"] = f"{task}_{treatment_method}_treated_{group}_trained_{trained_group}"
        hparams["bert_params"]["bert_state_dict"] = f"{pretrained_treated_model_dir}/pytorch_model.bin"
        logger.info(f"Treated Pretrained Model: {hparams['bert_params']['bert_state_dict']}")
        bert_treatment_test(model_ckpt, hparams, trainer, logger, task)


def predict_models(treatment="adjectives", trained_group="F",
                   pretrained_masking_method="double_num_adj", pretrained_epoch=0, pretrained_control=False,
                   sentiment_model_ckpt=None, ima_model_ckpt=None, pos_tagging_model_ckpt=None,
                   bert_state_dict=None):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hparams = {
        "treatment": treatment,
        "data_path": SENTIMENT_ADJECTIVES_DATASETS_DIR,
        "bert_params": {
            "bert_state_dict": bert_state_dict
        }
    }
    OUTPUT_DIR = f"{SENTIMENT_EXPERIMENTS_DIR}/{hparams['treatment']}/COMPARE"
    trainer = Trainer(gpus=1 if DEVICE.type == "cuda" else 0,
                      default_save_path=OUTPUT_DIR,
                      show_progress_bar=True,
                      early_stop_callback=None)
    hparams["output_path"] = trainer.logger.experiment.log_dir.rstrip('tf')
    logger = init_logger(f"testing", hparams["output_path"])
    for task, model in zip(("Sentiment", "CONTROL_IMA", "CONTROL_POS_Tagging"),
                           (sentiment_model_ckpt, ima_model_ckpt, pos_tagging_model_ckpt)):
        for group in ("F", "CF"):
            predict_models_unit(task, trained_group, group, model,
                                hparams, trainer, logger, pretrained_masking_method,
                                pretrained_epoch, pretrained_control, bert_state_dict)
    handler = GoogleDriveHandler()
    push_message = handler.push_files(hparams["output_path"])
    logger.info(push_message)


def predict_all_models(treatment: str, trained_group: str, masking_method: str, pretrained_epoch: int, pretrained_control: bool):

    predict_models(treatment, trained_group, masking_method, pretrained_epoch, pretrained_control)

    if pretrained_control:
        pretrained_treated_model_dir = f"{SENTIMENT_ADJECTIVES_PRETRAIN_IMA_DIR}/{masking_method}/model_control"
    else:
        pretrained_treated_model_dir = f"{SENTIMENT_ADJECTIVES_PRETRAIN_IMA_DIR}/{masking_method}/model"

    if pretrained_epoch is not None:
        pretrained_treated_model_dir = f"{pretrained_treated_model_dir}/epoch_{pretrained_epoch}"

    bert_state_dict = f"{pretrained_treated_model_dir}/pytorch_model.bin"
    if pretrained_control:
        trained_group = f"{trained_group}_ima_control_treated"
    else:
        trained_group = f"{trained_group}_ima_treated"
    predict_models(treatment, trained_group, masking_method, pretrained_epoch, pretrained_control, bert_state_dict=bert_state_dict)


if __name__ == "__main__":
    main()
