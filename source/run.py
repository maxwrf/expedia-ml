from source.data.data import Data
from source.data.features import Features


def run(config):
    d = Data(config)
    d.download_data()
    d.load_data()
    f = Features(config)

    if config.get('Data', 'remove_after_run') == 'True':
        d.remove_data()
