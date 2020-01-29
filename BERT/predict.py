from typing import Dict
from constants import SENTIMENT_EXPERIMENTS_DIR, SENTIMENT_IMA_DATA_DIR, SENTIMENT_MLM_DATA_DIR, POMS_MLM_DATA_DIR, POMS_GENDER_DATA_DIR, POMS_EXPERIMENTS_DIR
from pytorch_lightning import Trainer, LightningModule
from BERT.networks import LightningBertPretrainedClassifier, BertPretrainedClassifier
from os import listdir
from Timer import timer
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
def test_gender_models(factual_model_ckpt=None, counterfactual_model_ckpt=None):
    TREATMENT = "gender"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    HYPERPARAMETERS = {"bert_params": {}}
    # Factual OOB BERT Model training
    OUTPUT_DIR = f"{POMS_EXPERIMENTS_DIR}/{TREATMENT}/COMPARE"
    trainer = Trainer(gpus=1 if DEVICE.type == "cuda" else 0,
                      default_save_path=OUTPUT_DIR,
                      show_progress_bar=True,
                      early_stop_callback=None)
    HYPERPARAMETERS["output_path"] = trainer.logger.experiment.log_dir.rstrip('tf')
    if not factual_model_ckpt:
        factual_model_ckpt = f"{POMS_EXPERIMENTS_DIR}/{TREATMENT}/OOB_F/best_model/checkpoints"
    HYPERPARAMETERS["text_column"] = "Sentence_f"
    HYPERPARAMETERS["bert_params"]["name"] = "OOB_F"
    HYPERPARAMETERS["bert_params"]["bert_state_dict"] = None
    bert_treatment_test(factual_model_ckpt, HYPERPARAMETERS, trainer)
    # Factual OOB BERT Model test with MLM LM
    HYPERPARAMETERS["bert_params"]["name"] = "MLM"
    HYPERPARAMETERS["bert_params"]["bert_state_dict"] = f"{POMS_MLM_DATA_DIR}/model/pytorch_model.bin"
    bert_treatment_test(factual_model_ckpt, HYPERPARAMETERS, trainer)
    # Factual OOB BERT Model test with IMA LM
    HYPERPARAMETERS["bert_params"]["name"] = "Gender"
    HYPERPARAMETERS["bert_params"]["bert_state_dict"] = f"{POMS_GENDER_DATA_DIR}/model/pytorch_model.bin"
    bert_treatment_test(factual_model_ckpt, HYPERPARAMETERS, trainer)
    # CounterFactual OOB BERT Model training
    HYPERPARAMETERS["text_column"] = "Sentence_m"
    HYPERPARAMETERS["bert_params"]["name"] = "OOB_M"
    HYPERPARAMETERS["bert_params"]["bert_state_dict"] = None
    if not counterfactual_model_ckpt:
        counterfactual_model_ckpt = f"{POMS_EXPERIMENTS_DIR}/{TREATMENT}/OOB_M/best_model/checkpoints"
    bert_treatment_test(counterfactual_model_ckpt, HYPERPARAMETERS, trainer)


if __name__ == "__main__":
    test_gender_models()
