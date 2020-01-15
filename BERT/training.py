from constants import SENTIMENT_RAW_DATA_DIR, SENTIMENT_EXPERIMENTS_DIR, SENTIMENT_IMA_DATA_DIR, SENTIMENT_MLM_DATA_DIR
from pytorch_lightning import Trainer
from BERT.networks import LightningBertPretrainedClassifier, LightningHyperparameters, BertPretrainedClassifier
from utils import send_email, get_free_gpu, init_logger
from os import listdir
from Timer import timer
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
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
### Constants
PAD_ID = 0
BATCH_SIZE = 8
DROPOUT = 0.1
EPOCHS = 100
FP16 = False

HYPERPARAMETERS = {
    "data_path": DATASET_DIR,
    "treatment": TREATMENT,
    "text_column": TEXT_COLUMN,
    "label_column": LABEL_COLUMN,
    "bert_params": {
        "device": DEVICE,
        "batch_size": BATCH_SIZE,
        "dropout": DROPOUT,
        "bert_state_dict": BERT_STATE_DICT,
        "name": MODE
    }
}


def get_checkpoint_file(ckpt_dir):
    for file in sorted(listdir(ckpt_dir)):
        if file.endswith(".ckpt"):
            return f"{ckpt_dir}/{file}"
    else:
        return None


def print_final_metrics(metrics):
    print("\nFinal Metrics:")
    for metric, val in metrics.items():
        print(f"{metric}: {val:.4f}")


@timer
def bert_train_eval(hparams, output_dir):
    print(f"Training for {EPOCHS} epochs")
    trainer = Trainer(gpus=1 if DEVICE.type == "cuda" else 0,
                      default_save_path=output_dir,
                      show_progress_bar=True,
                      accumulate_grad_batches=8,
                      max_nb_epochs=EPOCHS,
                      early_stop_callback=None)
    hparams["output_path"] = trainer.logger.experiment.log_dir.rstrip('tf')
    model = LightningBertPretrainedClassifier(LightningHyperparameters(hparams))
    trainer.fit(model)
    trainer.test()
    print_final_metrics(trainer.tqdm_metrics)
    # best_model = LightningBertPretrainedClassifier.load_from_checkpoint(get_checkpoint_file(f"{model.hparams.output_path}/checkpoints/"))
    # best_model.eval()
    # best_model.freeze()
    # trainer.test(best_model)
    # print_final_metrics(trainer.tqdm_metrics)
    return model


@timer
def bert_treatment_test(model, hparams, output_dir):
    print(f"Testing model with {hparams['bert_params']['name']} LM")
    trainer = Trainer(gpus=1 if DEVICE.type == "cuda" else 0,
                      default_save_path=output_dir,
                      show_progress_bar=True,
                      accumulate_grad_batches=8,
                      max_nb_epochs=EPOCHS,
                      early_stop_callback=None)
    hparams["output_path"] = trainer.logger.experiment.log_dir.rstrip('tf')
    # model = LightningBertPretrainedClassifier(LightningHyperparameters(hparams))
    model.bert_classifier.bert = BertPretrainedClassifier.load_frozen_bert(model.bert_classifier.bert_pretrained_model,
                                                                           hparams["bert_params"]["bert_state_dict"])
    model.freeze()
    trainer.test(model)
    print_final_metrics(trainer.tqdm_metrics)


@timer
def main():
    # Factual OOB BERT Model training
    OUTPUT_DIR = f"{SENTIMENT_EXPERIMENTS_DIR}/{TREATMENT}/{DOMAIN}/OOB_F"
    factual_oob_model = bert_train_eval(HYPERPARAMETERS, OUTPUT_DIR)
    # CounterFactual OOB BERT Model training
    OUTPUT_DIR = f"{SENTIMENT_EXPERIMENTS_DIR}/{TREATMENT}/{DOMAIN}/OOB_CF"
    HYPERPARAMETERS["text_column"] = "no_adj_review"
    counterfactual_oob_model = bert_train_eval(HYPERPARAMETERS, OUTPUT_DIR)
    # Factual OOB BERT Model test with MLM LM
    OUTPUT_DIR = f"{SENTIMENT_EXPERIMENTS_DIR}/{TREATMENT}/{DOMAIN}/MLM"
    HYPERPARAMETERS["text_column"] = "review"
    HYPERPARAMETERS["bert_params"]["name"] = "MLM"
    HYPERPARAMETERS["bert_params"]["bert_state_dict"] = f"{SENTIMENT_MLM_DATA_DIR}/{DOMAIN}/model/pytorch_model.bin"
    bert_treatment_test(factual_oob_model, HYPERPARAMETERS, OUTPUT_DIR)
    # Factual OOB BERT Model test with IMA LM
    OUTPUT_DIR = f"{SENTIMENT_EXPERIMENTS_DIR}/{TREATMENT}/{DOMAIN}/IMA"
    HYPERPARAMETERS["bert_params"]["name"] = "IMA"
    HYPERPARAMETERS["bert_params"]["bert_state_dict"] = f"{SENTIMENT_IMA_DATA_DIR}/{DOMAIN}/model/pytorch_model.bin"
    bert_treatment_test(factual_oob_model, HYPERPARAMETERS, OUTPUT_DIR)


if __name__ == "__main__":
    main()
