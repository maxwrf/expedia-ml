from time import time
from datetime import timedelta
from source.run import run
import configparser
from pathlib import Path


def main():
    start = time()

    # Load configs
    config = configparser.ConfigParser()
    userconfig = 'Config/config.ini'
    userConfig = Path(userconfig)
    config.read(userConfig)

    # run the main process
    run(config)

    end = time()
    delta = (timedelta(end - start))
    print(delta)


if __name__ == "__main__":
    main()
