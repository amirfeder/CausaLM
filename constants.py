from os import getenv

HOME_DIR = getenv('HOME', "/home/{}".format(getenv('USER', "/home/amirf")))
DATA_DIR = f"{HOME_DIR}/GoogleDrive/AmirNadav/CausaLM/Data"
SENTIMENT_DATA_DIR = f"{DATA_DIR}/Sentiment"
SENTIMENT_RAW_DATA_DIR = f"{SENTIMENT_DATA_DIR}/Raw"
AMAZON_DATA_DIR = f"{DATA_DIR}/Amazon"
MOVIES_DATASET = f"{SENTIMENT_DATA_DIR}/movies/movie_data"
IMA_DATA_DIR = f"{DATA_DIR}/Sentiment/IMA"
SENTIMENT_DOMAINS = ["books", "electronics", "kitchen", "dvd"]
BERT_PRETRAINED_MODEL = 'bert-base-cased'
MAX_SEQ_LENGTH = 384
RANDOM_SEED = 212
