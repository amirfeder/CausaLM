from os import getenv

HOME_DIR = getenv('HOME', "/home/{}".format(getenv('USER', "/home/amirf")))
DATA_DIR = f"{HOME_DIR}/GoogleDrive/AmirNadav/CausaLM/Data"
SENTIMENT_DATA_DIR = f"{DATA_DIR}/Sentiment"
SENTIMENT_RAW_DATA_DIR = f"{SENTIMENT_DATA_DIR}/Raw"
AMAZON_DATA_DIR = f"{DATA_DIR}/Amazon"
MOVIES_DATASET = f"{SENTIMENT_DATA_DIR}/movies/movie_data"
SENTIMENT_DOMAINS = ["books", "electronics", "kitchen", "dvd", "movies"]
SENTIMENT_MODES = ["IMA", "MLM", "OOB"]
DOMAIN = "books"
MODE = "IMA"
BERT_PRETRAINED_MODEL = 'bert-base-cased'
MAX_SEQ_LENGTH = 384
RANDOM_SEED = 212
SENTIMENT_MODE_DATA_DIR = f"{DATA_DIR}/Sentiment/{MODE}"
SENTIMENT_IMA_DATA_DIR = f"{DATA_DIR}/Sentiment/IMA"
FINAL_PRETRAINED_MODEL = f"{SENTIMENT_MODE_DATA_DIR}/{DOMAIN}/model/pytorch_model.bin"
OOB_PRETRAINED_MODEL = BERT_PRETRAINED_MODEL
if MODE == "OOB":
    FINAL_PRETRAINED_MODEL = OOB_PRETRAINED_MODEL