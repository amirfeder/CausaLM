from os import getenv

HOME_DIR = getenv('HOME', "/home/{}".format(getenv('USER', "/home/amirf")))
DATA_DIR = f"{HOME_DIR}/GoogleDrive/AmirNadav/CausaLM/Data"
SENTIMENT_DATA_DIR = f"{DATA_DIR}/Sentiment"
IMA_DATA_DIR = f"{DATA_DIR}/Sentiment/IMA"
SENTIMENT_DOMAINS = ["books", "electronics", "kitchen", "dvd"]
BERT_PRETRAINED_MODEL = 'bert-base-cased'
MAX_SEQ_LENGTH = 512
