import time
from pathlib import Path
import configparser
from source.run import run
import warnings
from source.utilities.configure_logger import configure_logger


def main():
    warnings.simplefilter("ignore")
    start_time = time.time()

    # Load configs
    config = configparser.ConfigParser()
    userconfig = 'Config/config.ini'
    userConfig = Path(userconfig)
    config.read(userConfig)

    # configure logger
    configure_logger(config)

    # run the main process
    run(config)

    # print execution time
    print("--- %s minutes ---" % ((time.time() - start_time) / 60))


if __name__ == "__main__":
    main()
