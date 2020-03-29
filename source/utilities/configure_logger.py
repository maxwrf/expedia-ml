import logging


def configure_logger(config):
    logger = logging.getLogger('pipeline')
    logger.setLevel(logging.INFO)

    logFile = config.get('Logging', 'filename')
    fh = logging.FileHandler(logFile)
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    fh.setLevel(logging.INFO)
    ch.setLevel(logging.INFO)

    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
