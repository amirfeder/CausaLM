from typing import Dict
from constants import SENTIMENT_EXPERIMENTS_DIR, SENTIMENT_IMA_DATA_DIR, SENTIMENT_MLM_DATA_DIR, POMS_MLM_DATA_DIR, POMS_GENDER_DATA_DIR, POMS_RACE_DATA_DIR, POMS_EXPERIMENTS_DIR
from pytorch_lightning import Trainer, LightningModule
from BERT.networks import LightningBertPretrainedClassifier, BertPretrainedClassifier
from os import listdir
from Timer import timer
from utils import GoogleDriveHandler
import torch

# LOGGER = init_logger("OOB_training")


def get_checkpoint_file(ckpt_dir):
    for file in sorted(listdir(ckpt_dir)):
        if file.endswith(".ckpt"):
            return f"{ckpt_dir}/{file}"
    else:
        return None


def print_final_metrics(name: str, metrics: Dict):
    print(f"{name} Metrics:")
    for metric, val in metrics.items():
        print(f"{metric}: {val:.4f}")
    print()


@timer
def bert_treatment_test(model_ckpt, hparams, trainer):
    print(f"Testing model with {hparams['bert_params']['name']} LM")
    if isinstance(model_ckpt, LightningModule):
        model = model_ckpt
    else:
        model_ckpt_file = get_checkpoint_file(model_ckpt)
        model = LightningBertPretrainedClassifier.load_from_checkpoint(model_ckpt_file)
    model.hparams.output_path = hparams["output_path"]
    if hparams["bert_params"]["bert_state_dict"]:
        model.bert_classifier.bert = BertPretrainedClassifier.load_frozen_bert(model.bert_classifier.bert_pretrained_model,
                                                                               hparams["bert_params"]["bert_state_dict"])
        model.bert_classifier.name = f"{hparams['bert_params']['name']}"
    model.freeze()
    trainer.test(model)
    print_final_metrics(hparams['bert_params']['name'], trainer.tqdm_metrics)


@timer
def test_adj_models(factual_model_ckpt=None, counterfactual_model_ckpt=None):
    DOMAIN = "movies"
    TREATMENT = "adj"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    HYPERPARAMETERS = {"bert_params": {}}
    # Factual OOB BERT Model training
    OUTPUT_DIR = f"{SENTIMENT_EXPERIMENTS_DIR}/{TREATMENT}/{DOMAIN}/COMPARE"
    trainer = Trainer(gpus=1 if DEVICE.type == "cuda" else 0,
                      default_save_path=OUTPUT_DIR,
                      show_progress_bar=True,
                      early_stop_callback=None)
    HYPERPARAMETERS["output_path"] = trainer.logger.experiment.log_dir.rstrip('tf')
    if not factual_model_ckpt:
        factual_model_ckpt = f"{SENTIMENT_EXPERIMENTS_DIR}/{TREATMENT}/{DOMAIN}/OOB_F/best_model/checkpoints"
    HYPERPARAMETERS["text_column"] = "review"
    HYPERPARAMETERS["bert_params"]["name"] = "OOB_F"
    HYPERPARAMETERS["bert_params"]["bert_state_dict"] = None
    bert_treatment_test(factual_model_ckpt, HYPERPARAMETERS, trainer)
    # Factual OOB BERT Model test with MLM LM
    HYPERPARAMETERS["bert_params"]["name"] = "MLM"
    HYPERPARAMETERS["bert_params"]["bert_state_dict"] = f"{SENTIMENT_MLM_DATA_DIR}/{DOMAIN}/model/pytorch_model.bin"
    bert_treatment_test(factual_model_ckpt, HYPERPARAMETERS, trainer)
    # Factual OOB BERT Model test with IMA LM
    HYPERPARAMETERS["bert_params"]["name"] = "IMA"
    HYPERPARAMETERS["bert_params"]["bert_state_dict"] = f"{SENTIMENT_IMA_DATA_DIR}/{DOMAIN}/model/pytorch_model.bin"
    bert_treatment_test(factual_model_ckpt, HYPERPARAMETERS, trainer)
    # Factual OOB BERT Model test with MLM LM (double_samples)
    HYPERPARAMETERS["bert_params"]["name"] = "MLM_double_samples"
    HYPERPARAMETERS["bert_params"]["bert_state_dict"] = f"{SENTIMENT_MLM_DATA_DIR}/double/{DOMAIN}/model/pytorch_model.bin"
    bert_treatment_test(factual_model_ckpt, HYPERPARAMETERS, trainer)
    # Factual OOB BERT Model test with IMA LM (double_adj)
    HYPERPARAMETERS["bert_params"]["name"] = "IMA_double_adj"
    HYPERPARAMETERS["bert_params"]["bert_state_dict"] = f"{SENTIMENT_IMA_DATA_DIR}/double/{DOMAIN}/model/pytorch_model.bin"
    bert_treatment_test(factual_model_ckpt, HYPERPARAMETERS, trainer)
    # CounterFactual OOB BERT Model training
    HYPERPARAMETERS["text_column"] = "no_adj_review"
    HYPERPARAMETERS["bert_params"]["name"] = "OOB_CF"
    HYPERPARAMETERS["bert_params"]["bert_state_dict"] = None
    if not counterfactual_model_ckpt:
        counterfactual_model_ckpt = f"{SENTIMENT_EXPERIMENTS_DIR}/{TREATMENT}/{DOMAIN}/OOB_CF/best_model/checkpoints"
    bert_treatment_test(counterfactual_model_ckpt, HYPERPARAMETERS, trainer)


@timer
def test_genderace_models(treatment="gender", factual_poms_model_ckpt=None, counterfactual_poms_model_ckpt=None,
                          factual_control_model_ckpt=None, counterfactual_control_model_ckpt=None):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    HYPERPARAMETERS = {"bert_params": {}}
    OUTPUT_DIR = f"{POMS_EXPERIMENTS_DIR}/{treatment}/COMPARE"
    trainer = Trainer(gpus=1 if DEVICE.type == "cuda" else 0,
                      default_save_path=OUTPUT_DIR,
                      show_progress_bar=True,
                      early_stop_callback=None)
    HYPERPARAMETERS["output_path"] = trainer.logger.experiment.log_dir.rstrip('tf')
    if "enriched" in treatment:
        state_dict_dir = "model_enriched"
    else:
        state_dict_dir = "model"
    if "gender" in treatment:
        TREATMENT = "Gender"
        pretrained_treated_model_dir = f"{POMS_GENDER_DATA_DIR}/{state_dict_dir}"
    else:
        TREATMENT = "Race"
        pretrained_treated_model_dir = f"{POMS_RACE_DATA_DIR}/{state_dict_dir}"
    # Factual POMS BERT Model training
    HYPERPARAMETERS["text_column"] = "Sentence_F"
    HYPERPARAMETERS["bert_params"]["name"] = "POMS_F"
    HYPERPARAMETERS["bert_params"]["label_size"] = 5
    if not factual_poms_model_ckpt:
        factual_poms_model_ckpt = f"{POMS_EXPERIMENTS_DIR}/{treatment}/{HYPERPARAMETERS['bert_params']['name']}/best_model/checkpoints"
    HYPERPARAMETERS["bert_params"]["bert_state_dict"] = None
    bert_treatment_test(factual_poms_model_ckpt, HYPERPARAMETERS, trainer)
    # Factual POMS BERT Model test with MLM LM
    HYPERPARAMETERS["bert_params"]["name"] = "POMS_MLM"
    HYPERPARAMETERS["bert_params"]["bert_state_dict"] = f"{POMS_MLM_DATA_DIR}/{state_dict_dir}/pytorch_model.bin"
    bert_treatment_test(factual_poms_model_ckpt, HYPERPARAMETERS, trainer)
    # Factual POMS BERT Model test with Gender/Race LM
    HYPERPARAMETERS["bert_params"]["name"] = f"POMS_{TREATMENT}"
    HYPERPARAMETERS["bert_params"]["bert_state_dict"] = f"{pretrained_treated_model_dir}/pytorch_model.bin"
    bert_treatment_test(factual_poms_model_ckpt, HYPERPARAMETERS, trainer)
    # CounterFactual POMS BERT Model training
    HYPERPARAMETERS["text_column"] = "Sentence_CF"
    HYPERPARAMETERS["bert_params"]["name"] = "POMS_CF"
    HYPERPARAMETERS["bert_params"]["bert_state_dict"] = None
    if not counterfactual_poms_model_ckpt:
        counterfactual_poms_model_ckpt = f"{POMS_EXPERIMENTS_DIR}/{treatment}/{HYPERPARAMETERS['bert_params']['name']}/best_model/checkpoints"
    bert_treatment_test(counterfactual_poms_model_ckpt, HYPERPARAMETERS, trainer)
    # Factual CONTROL BERT Model training
    HYPERPARAMETERS["label_column"] = "Gender_F"
    HYPERPARAMETERS["text_column"] = "Sentence_F"
    HYPERPARAMETERS["bert_params"]["name"] = "CONTROL_F"
    HYPERPARAMETERS["bert_params"]["label_size"] = 2
    if not factual_control_model_ckpt:
        factual_control_model_ckpt = f"{POMS_EXPERIMENTS_DIR}/{treatment}/{HYPERPARAMETERS['bert_params']['name']}/best_model/checkpoints"
    HYPERPARAMETERS["bert_params"]["bert_state_dict"] = None
    bert_treatment_test(factual_control_model_ckpt, HYPERPARAMETERS, trainer)
    # Factual CONTROL BERT Model test with MLM LM
    HYPERPARAMETERS["bert_params"]["name"] = "CONTROL_MLM"
    HYPERPARAMETERS["bert_params"]["bert_state_dict"] = f"{POMS_MLM_DATA_DIR}/{state_dict_dir}/pytorch_model.bin"
    bert_treatment_test(factual_control_model_ckpt, HYPERPARAMETERS, trainer)
    # Factual CONTROL BERT Model test with Gender/Race LM
    HYPERPARAMETERS["bert_params"]["name"] = f"CONTROL_{TREATMENT}"
    HYPERPARAMETERS["bert_params"]["bert_state_dict"] = f"{pretrained_treated_model_dir}/pytorch_model.bin"
    bert_treatment_test(factual_control_model_ckpt, HYPERPARAMETERS, trainer)
    # CounterFactual CONTROL BERT Model training
    HYPERPARAMETERS["label_column"] = "Gender_CF"
    HYPERPARAMETERS["text_column"] = "Sentence_CF"
    HYPERPARAMETERS["bert_params"]["name"] = "CONTROL_CF"
    HYPERPARAMETERS["bert_params"]["bert_state_dict"] = None
    if not counterfactual_control_model_ckpt:
        counterfactual_control_model_ckpt = f"{POMS_EXPERIMENTS_DIR}/{treatment}/{HYPERPARAMETERS['bert_params']['name']}/best_model/checkpoints"
    bert_treatment_test(counterfactual_control_model_ckpt, HYPERPARAMETERS, trainer)
    push_results_to_google_drive(HYPERPARAMETERS["output_path"])


def push_results_to_google_drive(path: str):
    try:
        handler = GoogleDriveHandler()
        push_return = handler.push_files(path)
        if push_return[0] == 0:
            print(f"Successfully pushed results to Google Drive: {path}")
        else:
            print(f"Failed to push results to Google Drive: {path}")
            print(f"Exit Code: {push_return[0]}\nSTDOUT: {push_return[1]}\nSTDERR: {push_return[2]}")
    except Exception as e:
        print(f"ERROR: {e}")
        print(f"Failed to push results to Google Drive: {path}")


if __name__ == "__main__":
    test_genderace_models()
