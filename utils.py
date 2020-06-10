from os import listdir, path
from glob import glob
from datetime import datetime
from subprocess import Popen, PIPE, run
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict
import pandas as pd
import torch
import logging

HOME_DIR = str(Path.home())
INIT_TIME = datetime.now().strftime('%e-%m-%y_%H-%M-%S').lstrip()


def init_logger(name=None, path=None, screen=True):
    if name is None:
        name = __name__
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('{asctime} - {message}', datefmt="%H:%M:%S", style="{")
    if path:
        Path(path).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(f"{path}/{name}-{INIT_TIME}.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    if screen:
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


def save_predictions(folder, sample_idx_list, predictions_list, true_list, correct_list, class_probs, name):
    df_dict = {
        "sample_index": sample_idx_list,
        "prediction": predictions_list,
        "true": true_list,
        "correct": correct_list,
    }
    df_dict.update({f"class_{i}_prob": class_i_prob for i, class_i_prob in enumerate(class_probs)})
    df = pd.DataFrame.from_dict(df_dict)
    df = df.set_index("sample_index").sort_index()
    df.to_csv(f"{folder}/{name}-predictions.csv")


class GoogleDriveHandler:
    def __init__(self,
                 local_root: str = f"{HOME_DIR}/GoogleDrive",
                 drive_binary: str = f"{HOME_DIR}/bin/go/packages/bin/drive",
                 default_timeout: int = 600):
        self.local_root = local_root
        self.drive_binary = drive_binary
        self.default_args = ["-no-prompt"]
        self.default_timeout = default_timeout

    def _execute_drive_cmd(self, subcommand: str, path: str, cmd_args: list):
        if subcommand not in ("pull", "push"):
            raise ValueError("Only pull and push commands are currently supported")
        cmd = [self.drive_binary, subcommand] + self.default_args + cmd_args + [path]
        cmd_return = run(cmd, capture_output=True, text=True, timeout=self.default_timeout, cwd=HOME_DIR)
        return cmd_return.returncode, cmd_return.stdout, cmd_return.stderr

    def push_files(self, path: str, cmd_args: list = []):
        try:
            push_return = self._execute_drive_cmd("push", path, ["-files"] + cmd_args)
            if push_return[0] == 0:
                message = f"Successfully pushed results to Google Drive: {path}"
            else:
                message = f"Failed to push results to Google Drive: {path}\nExit Code: {push_return[0]}\nSTDOUT: {push_return[1]}\nSTDERR: {push_return[2]}"
        except Exception as e:
            message = f"ERROR: {e}\nFailed to push results to Google Drive: {path}"
        return message

    def pull_files(self, path: str, cmd_args: list = []):
        return self._execute_drive_cmd("pull", path, ["-files"] + cmd_args)


def get_checkpoint_file(ckpt_dir: str):
    for file in sorted(listdir(ckpt_dir)):
        if file.endswith(".ckpt"):
            return f"{ckpt_dir}/{file}"
    else:
        return None


def find_latest_model_checkpoint(models_dir: str):
    model_ckpt = None
    while not model_ckpt:
        model_versions = sorted(glob(models_dir), key=path.getctime)
        if model_versions:
            latest_model = model_versions.pop()
            model_ckpt_dir = f"{latest_model}/checkpoints"
            model_ckpt = get_checkpoint_file(model_ckpt_dir)
        else:
            raise FileNotFoundError(f"Couldn't find a model checkpoint in {models_dir}")
    return model_ckpt


def print_final_metrics(name: str, metrics: Dict, logger=None):
    if logger:
        logger.info(f"{name} Metrics:")
        for metric, val in metrics.items():
            logger.info(f"{metric}: {val:.4f}")
        logger.info("\n")
    else:
        print(f"{name} Metrics:")
        for metric, val in metrics.items():
            print(f"{metric}: {val:.4f}")
        print()
