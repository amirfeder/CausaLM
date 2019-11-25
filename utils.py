import logging
from os import getenv

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
