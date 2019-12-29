from constants import SENTIMENT_MODE_DATA_DIR, SENTIMENT_RAW_DATA_DIR, DOMAIN, FINAL_PRETRAINED_MODEL
from pytorch_lightning import Trainer
from networks import LightningBertPretrainedClassifier
from utils import send_email, get_free_gpu
import torch

DATASET_DIR = f"{SENTIMENT_RAW_DATA_DIR}/{DOMAIN}"
OUTPUT_DIR = f"{SENTIMENT_MODE_DATA_DIR}/{DOMAIN}"
DEVICE = get_free_gpu()
### Constants
PAD_ID = 0
BATCH_SIZE = 8
DROPOUT = 0.1
EPOCHS = 100
FP16 = False


def main():
    oob_model = LightningBertPretrainedClassifier(DATASET_DIR, DEVICE, BATCH_SIZE, DROPOUT)
    trainer = Trainer(fast_dev_run=True, overfit_pct=0.1,
                      gpus=[DEVICE.index if DEVICE.type == "cuda" else None],
                      default_save_path=OUTPUT_DIR,
                      show_progress_bar=True,
                      accumulate_grad_batches=8,
                      max_nb_epochs=EPOCHS)
    trainer.fit(oob_model)


if __name__ == "__main__":
    main()
