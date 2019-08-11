from classifiers.player_performance.data_handlers2 import all_pbp_game_tokenized_text_data
from constants import resources_dir

FINE_TUNE_DATA_DIR = f"{resources_dir}BERT_Fine-tune/"
BERT_QA_FINE_TUNE_DATA_DIR = f"{resources_dir}BERT_Fine-tune/QA/"
ENCODINGS_DIR = f"{resources_dir}BERT_Encodings/"
BERT_QA_ENCODINGS_DIR = f"{ENCODINGS_DIR}/QA/"
BERT_SENTENCE_ENCODINGS_DIR = f"{ENCODINGS_DIR}Sentences/"
BERT_PRETRAINED_MODEL = 'bert-base-uncased'
MAX_QA_SEQ_LENGTH = 512
MAX_SENTENCE_SEQ_LENGTH = 384
DATASET_VERSION = "v7l3"
QA_UNIQUE_ID_MAPPING = f"BERT_QA_ID-{DATASET_VERSION}"
ALL_PBP_GAME_TEXT_METRICS = all_pbp_game_tokenized_text_data.replace("v7l3.csv", f"{DATASET_VERSION}.csv").replace("_tokenized", "")
ALL_PBP_PERIOD_TEXT_METRICS = ALL_PBP_GAME_TEXT_METRICS.replace("game", "period")