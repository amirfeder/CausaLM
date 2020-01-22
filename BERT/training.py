from constants import SENTIMENT_RAW_DATA_DIR, SENTIMENT_EXPERIMENTS_DIR
from pytorch_lightning import Trainer
from BERT.networks import LightningBertPretrainedClassifier, LightningHyperparameters
from BERT.predict import test_models, print_final_metrics
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
BATCH_SIZE = 128
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
        "device": DEVICE,
        "batch_size": BATCH_SIZE,
        "dropout": DROPOUT,
        "bert_state_dict": BERT_STATE_DICT,
        "name": MODE
    }
}


@timer
def bert_train_eval(hparams, output_dir):
    print(f"Training for {EPOCHS} epochs")
    trainer = Trainer(gpus=1 if DEVICE.type == "cuda" else 0,
                      default_save_path=output_dir,
                      show_progress_bar=True,
                      accumulate_grad_batches=ACCUMULATE,
                      max_nb_epochs=EPOCHS,
                      early_stop_callback=None)
    hparams["output_path"] = trainer.logger.experiment.log_dir.rstrip('tf')
    model = LightningBertPretrainedClassifier(LightningHyperparameters(hparams))
    trainer.fit(model)
    trainer.test()
    print_final_metrics(hparams['bert_params']['name'], trainer.tqdm_metrics)
    return model


@timer
def main():
    # Factual OOB BERT Model training
    OUTPUT_DIR = f"{SENTIMENT_EXPERIMENTS_DIR}/{TREATMENT}/{DOMAIN}/OOB_F"
    factual_oob_model = bert_train_eval(HYPERPARAMETERS, OUTPUT_DIR)
    # CounterFactual OOB BERT Model training
    OUTPUT_DIR = f"{SENTIMENT_EXPERIMENTS_DIR}/{TREATMENT}/{DOMAIN}/OOB_CF"
    HYPERPARAMETERS["text_column"] = "no_adj_review"
    HYPERPARAMETERS["bert_params"]["name"] = "OOB_CF"
    counterfactual_oob_model = bert_train_eval(HYPERPARAMETERS, OUTPUT_DIR)
    test_models(factual_oob_model, counterfactual_oob_model)


if __name__ == "__main__":
    main()
