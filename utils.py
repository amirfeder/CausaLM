from datetime import datetime
from os import getenv
from smtplib import SMTP
from os import environ
from email.message import EmailMessage
from subprocess import Popen, PIPE
from multiprocessing import cpu_count
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import logging
import sys

INIT_TIME = datetime.now().strftime('%e-%m-%y_%H-%M-%S').lstrip()
HOME_DIR = getenv('HOME', "/home/{}".format(getenv('USER', "/home/amirf")))
PROJECT_DIR = f"{HOME_DIR}/GoogleDrive/AmirNadav/CausaLM"


def init_logger(name=None, file=None):
    if name is None:
        name = __name__
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('{asctime} - {message}', datefmt="%H:%M:%S", style="{")
    if file:
        file = file.replace('.py', '.log')
        file_handler = logging.FileHandler(file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    screen_handler = logging.StreamHandler()
    screen_handler.setFormatter(formatter)
    logger.addHandler(screen_handler)
    return logger


def get_free_gpu():
    if torch.cuda.is_available():
        gpu_output = Popen(["nvidia-smi", "-q", "-d", "PIDS"], stdout=PIPE, text=True)
        gpu_processes = Popen(["grep", "Processes"], stdin=gpu_output.stdout, stdout=PIPE, text=True)
        gpu_output.stdout.close()
        processes_output = gpu_processes.communicate()[0]
        for i, line in enumerate(processes_output.strip().split("\n")):
            if line.endswith("None"):
                print(f"Found Free GPU ID: {i}")
                cuda_device = f"cuda:{i}"
                torch.cuda.set_device(cuda_device)
                return torch.device(cuda_device)
        print("WARN - No Free GPU found! Running on CPU instead...")
    return torch.device("cpu")


def count_num_cpu_gpu():
    if torch.cuda.is_available():
        num_gpu_cores = torch.cuda.device_count()
        num_cpu_cores = (cpu_count() // num_gpu_cores // 2) - 1
    else:
        num_gpu_cores = 0
        num_cpu_cores = (cpu_count() // 2) - 1
    return num_cpu_cores, num_gpu_cores


def send_email(message, module_name):
    """utility function to send an email with results from a training run"""
    message_string = '\n'.join(message)
    recipients = ['nadavo@campus.technion.ac.il', 'feder@campus.technion.ac.il']
    msg = EmailMessage()
    msg['Subject'] = f"Finished {module_name} on {environ['HOSTNAME']}"
    msg['From'] = 'someserver@technion.ac.il'
    msg['To'] = ', '.join(recipients)
    msg.set_content(message_string)
    sender = SMTP('localhost')
    sender.send_message(msg)
    sender.quit()

class StreamToLogger:
    """
   Fake file-like stream object that redirects writes to a logger instance.
   written by: Ferry Boender
   https://www.electricmonk.nl/log/2011/08/14/redirect-stdout-and-stderr-to-a-logger-in-python/
   """
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())


class PyTorchTrainer:
    init_time = INIT_TIME

    def __init__(self, name, logger=None, server_mode=None, save_path=None):
        self.name = name
        self.folder = Path(f"{save_path}{self.name}-{PyTorchTrainer.init_time}/")
        self.folder.mkdir(parents=True, exist_ok=True)
        self.model_file_name = self.folder / "model.pt"
        self.results_file_name = self.folder / "results.pkl"
        self.log_file_name = self.folder / "output.log"
        if logger is None:
            self.logger = self.init_logger(server_mode)
        else:
            self.logger = logger

    def init_logger(self, server_mode):
        global loggers
        existing_logger = loggers.get(self.name, False)
        if existing_logger:
            return existing_logger
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.INFO)
        file_formatter = logging.Formatter("{levelname}@{asctime} - {message}", datefmt="%H:%M:%S", style="{")
        console_formatter = logging.Formatter("{asctime} - {message}", datefmt="%H:%M:%S", style="{")
        fh = logging.FileHandler(self.log_file_name)
        fh.setFormatter(file_formatter)
        logger.addHandler(fh)
        if server_mode:
            sys.stderr = StreamToLogger(logger, logging.ERROR)
        else:
            sh = logging.StreamHandler()
            sh.setFormatter(console_formatter)
            logger.addHandler(sh)
        loggers[self.name] = logger
        return logger

    def print_or_log(self, message=""):
        if self.logger:
            self.logger.info(message)
        else:
            print(message)

    def save_plot(self, label, fold=1, iterables_dict={}):
        for metric, metric_dict in iterables_dict.items():
            for dataset, values in metric_dict.items():
                plt.plot(values, label=dataset)
            plt.xlabel("Epochs")
            plt.ylabel(metric)
            plt.title(f"{label} {metric} by Epochs for Fold {fold}")
            plt.legend()
            plt.savefig(self.folder / f"graph-{metric}-fold{fold}")
            plt.clf()

    def save_scores(self, scores_list, scores_dataset, scores_metric):
        scores_array = np.array(scores_list, dtype=np.float32)
        rounded_scores_array = scores_array.round(decimals=6)
        np.save(self.folder / f"{scores_dataset}-{scores_metric}-scores", rounded_scores_array)

    def save_predictions(self, sample_idx_list, predictions_list, true_list, dataset):
        df = pd.DataFrame.from_dict({"sample_index": sample_idx_list, "prediction": predictions_list, "true": true_list})
        df = df.set_index("sample_index")
        df.to_csv(self.folder / f"{dataset}-predictions.csv")


def save_predictions(folder, sample_idx_list, predictions_list, true_list, correct_list, dataset):
    df = pd.DataFrame.from_dict({"sample_index": sample_idx_list,
                                 "prediction": predictions_list,
                                 "true": true_list,
                                 "correct": correct_list})
    df = df.set_index("sample_index")
    df.to_csv(f"{folder}/{dataset}-predictions.csv")
