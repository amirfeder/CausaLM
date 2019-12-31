from constants import SENTIMENT_MODE_DATA_DIR, SENTIMENT_RAW_DATA_DIR, DOMAIN, FINAL_PRETRAINED_MODEL
from pytorch_lightning import Trainer
from networks import LightningBertPretrainedClassifier
from utils import send_email, get_free_gpu, init_logger
import torch

LOGGER = init_logger("OOB_training")
TREATMENT = "adj"
TEXT_COLUMN = "review"
LABEL_COLUMN = "label"
DATASET_DIR = f"{SENTIMENT_RAW_DATA_DIR}/{DOMAIN}"
OUTPUT_DIR = f"{SENTIMENT_MODE_DATA_DIR}/{DOMAIN}"
# DEVICE = get_free_gpu()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
### Constants
PAD_ID = 0
BATCH_SIZE = 8
DROPOUT = 0.1
EPOCHS = 100
FP16 = False


def main():
    oob_model = LightningBertPretrainedClassifier(DATASET_DIR, TREATMENT, TEXT_COLUMN, LABEL_COLUMN, DEVICE, BATCH_SIZE, DROPOUT)
    trainer = Trainer(fast_dev_run=True, overfit_pct=0.1,
                      gpus=1 if DEVICE.type == "cuda" else 0,
                      default_save_path=OUTPUT_DIR,
                      show_progress_bar=True,
                      accumulate_grad_batches=8,
                      max_nb_epochs=EPOCHS)
    trainer.fit(oob_model)


if __name__ == "__main__":
    main()
