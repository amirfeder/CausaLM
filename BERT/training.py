from constants import SENTIMENT_MODE_DATA_DIR, SENTIMENT_RAW_DATA_DIR, DOMAIN, FINAL_PRETRAINED_MODEL, SENTIMENT_EXPERIMENTS_DIR
from pytorch_lightning import Trainer
from networks import LightningBertPretrainedClassifier, LightningHyperparameters
from utils import send_email, get_free_gpu, init_logger
from os import listdir
from Timer import timer
import torch

# LOGGER = init_logger("OOB_training")
MODE = "OOB"
BERT_STATE_DICT = None
TREATMENT = "adj"
TEXT_COLUMN = "review"
LABEL_COLUMN = "label"
DATASET_DIR = f"{SENTIMENT_RAW_DATA_DIR}/{DOMAIN}"
OUTPUT_DIR = f"{SENTIMENT_EXPERIMENTS_DIR}/{TREATMENT}/{DOMAIN}/{MODE}"
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
        "bert_state_dict": BERT_STATE_DICT
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
def bert_train_eval(hparams):
    print(f"Training for {EPOCHS} epochs")
    trainer = Trainer(gpus=1 if DEVICE.type == "cuda" else 0,
                      default_save_path=OUTPUT_DIR,
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


if __name__ == "__main__":
    bert_train_eval(HYPERPARAMETERS)
