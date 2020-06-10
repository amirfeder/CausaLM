from argparse import ArgumentParser
from constants import POMS_MLM_DIR, POMS_GENDER_MODEL_DIR, POMS_RACE_MODEL_DIR, POMS_EXPERIMENTS_DIR, MAX_POMS_SEQ_LENGTH
from pytorch_lightning import Trainer, LightningModule
from BERT.bert_text_classifier import LightningBertPretrainedClassifier, BertPretrainedClassifier
from copy import deepcopy
from utils import GoogleDriveHandler,  init_logger, print_final_metrics, find_latest_model_checkpoint
import torch

def main():
    parser = ArgumentParser()
    parser.add_argument("--treatment", type=str, required=True, default="gender", choices=("gender", "race"),
                        help="Treatment variable")
    parser.add_argument("--corpus_type", type=str, required=False, default="")
    parser.add_argument("--trained_group", type=str, required=False, default="F",
                        help="Specify data group for trained_models: F (factual) or CF (counterfactual)")
    parser.add_argument("--pretrained_epoch", type=int, required=False, default=0,
                        help="Specify epoch for pretrained models: 0-4")
    args = parser.parse_args()

    predict_all_models(args.treatment, args.corpus_type, args.trained_group, args.pretrained_epoch)


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
    model.hparams.max_seq_len = hparams["max_seq_len"]
    model.hparams.output_path = hparams["output_path"]
    model.hparams.label_column = hparams["label_column"]
    model.hparams.text_column = hparams["text_column"]
    model.hparams.bert_params["name"] = hparams['bert_params']['name']
    model.hparams.bert_params["label_size"] = hparams["bert_params"]["label_size"]
    model.bert_classifier.name = hparams['bert_params']['name']
    model.bert_classifier.label_size = hparams["bert_params"]["label_size"]

    model.freeze()
    trainer.test(model)
    print_final_metrics(hparams['bert_params']['name'], trainer.tqdm_metrics, logger)


def predict_models_unit(task, treatment, trained_group, group, model_ckpt,
                        hparams, trainer, logger, pretrained_epoch, bert_state_dict):
    if "noisy" in treatment:
        state_dict_dir = "model_enriched_noisy"
    elif "enriched" in treatment:
        state_dict_dir = "model_enriched"
    else:
        state_dict_dir = "model"
    if pretrained_epoch is not None:
        state_dict_dir = f"{state_dict_dir}/epoch_{pretrained_epoch}"
    if treatment.startswith("gender"):
        TREATMENT = "gender"
        pretrained_treated_model_dir = f"{POMS_GENDER_MODEL_DIR}/{state_dict_dir}"
    else:
        TREATMENT = "race"
        pretrained_treated_model_dir = f"{POMS_RACE_MODEL_DIR}/{state_dict_dir}"
    label_size = 2
    if task == "POMS":
        label_column = f"{task}_label"
        label_size = 5
    elif task.split("_")[-1].lower() in treatment:
        label_column = f"{task.split('_')[-1]}_{group}_label"
    else:
        label_column = f"{task.split('_')[-1]}_label"

    hparams["max_seq_len"] = MAX_POMS_SEQ_LENGTH
    hparams["label_column"] = label_column
    hparams["bert_params"]["label_size"] = label_size
    hparams["text_column"] = f"Sentence_{group}"
    hparams["treatment"] = treatment
    hparams["trained_group"] = trained_group
    logger.info(f"Treatment: {treatment}")
    logger.info(f"Text Column: {hparams['text_column']}")
    logger.info(f"Label Column: {label_column}")
    logger.info(f"Label Size: {label_size}")
    hparams["bert_params"]["name"] = f"{task}_{group}_trained_{trained_group}"
    hparams["bert_params"]["bert_state_dict"] = bert_state_dict

    if not model_ckpt:
        model_name = f"{hparams['label_column'].split('_')[0]}_{hparams['trained_group']}"
        models_dir = f"{POMS_EXPERIMENTS_DIR}/{hparams['treatment']}/{model_name}/lightning_logs/*"
        model_ckpt = find_latest_model_checkpoint(models_dir)

    # Group Task BERT Model training
    logger.info(f"Model: {hparams['bert_params']['name']}")
    bert_treatment_test(model_ckpt, hparams, trainer, logger)

    # Group Task BERT Model test with MLM LM
    hparams["bert_params"]["name"] = f"{task}_MLM_{group}_trained_{trained_group}"
    hparams["bert_params"]["bert_state_dict"] = f"{POMS_MLM_DIR}/{state_dict_dir}/pytorch_model.bin"
    logger.info(f"MLM Pretrained Model: {POMS_MLM_DIR}/{state_dict_dir}/pytorch_model.bin")
    bert_treatment_test(model_ckpt, hparams, trainer, logger)

    if not bert_state_dict:
        # Group Task BERT Model test with Gender/Race treated LM
        hparams["bert_params"]["name"] = f"{task}_{TREATMENT}_treated_{group}_trained_{trained_group}"
        hparams["bert_params"]["bert_state_dict"] = f"{pretrained_treated_model_dir}/pytorch_model.bin"
        logger.info(f"Treated Pretrained Model: {pretrained_treated_model_dir}/pytorch_model.bin")
        bert_treatment_test(model_ckpt, hparams, trainer, logger)


def predict_models(treatment="gender", trained_group="F", pretrained_epoch=None,
                   poms_model_ckpt=None, gender_model_ckpt=None, race_model_ckpt=None,
                   bert_state_dict=None):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hparams = {
        "treatment": treatment,
        "bert_params": {
            "bert_state_dict": bert_state_dict
        }
    }
    OUTPUT_DIR = f"{POMS_EXPERIMENTS_DIR}/{treatment}/COMPARE"
    trainer = Trainer(gpus=1 if DEVICE.type == "cuda" else 0,
                      default_save_path=OUTPUT_DIR,
                      show_progress_bar=True,
                      early_stop_callback=None)
    hparams["output_path"] = trainer.logger.experiment.log_dir.rstrip('tf')
    logger = init_logger(f"testing", hparams["output_path"])
    for task, model in zip(("POMS", "CONTROL_Gender", "CONTROL_Race"),
                           (poms_model_ckpt, gender_model_ckpt, race_model_ckpt)):
        for group in ("F", "CF"):
            predict_models_unit(task, treatment, trained_group, group, model,
                                hparams, trainer, logger, pretrained_epoch, bert_state_dict)
    handler = GoogleDriveHandler()
    push_message = handler.push_files(hparams["output_path"])
    logger.info(push_message)


def predict_all_models(treatment: str, corpus_type: str, trained_group: str, pretrained_epoch: int = None):
    if corpus_type:
        treatment = f"{treatment}_{corpus_type}"
        state_dict_dir = f"model_{corpus_type}"
    else:
        state_dict_dir = "model"
    if pretrained_epoch is not None:
        state_dict_dir = f"{state_dict_dir}/epoch_{pretrained_epoch}"
    if treatment.startswith("gender"):
        pretrained_treated_model_dir = f"{POMS_GENDER_MODEL_DIR}/{state_dict_dir}"
    else:
        pretrained_treated_model_dir = f"{POMS_RACE_MODEL_DIR}/{state_dict_dir}"

    predict_models(treatment, trained_group, pretrained_epoch)

    bert_state_dict = f"{pretrained_treated_model_dir}/pytorch_model.bin"
    trained_group = f"{trained_group}_{treatment.split('_')[0]}_treated"
    predict_models(treatment, trained_group, pretrained_epoch, bert_state_dict=bert_state_dict)


if __name__ == "__main__":
    main()
