import os
import urllib.request
import zipfile
import gzip
import shutil
import pandas as pd


class Data():
    def __init__(self, config):
        self.project_path = config.get('General', 'path')
        self.data_path = self.project_path + '/data/'
        self.url = config.get('Data', 'url')
        self.csvs = config.get('Data', 'csvs').split(', ')
        self.gzs = config.get('Data', 'gzs').split(', ')
        self.is_development = 'True' == config.get('General', 'is_development')
        self.write_sample_files = 'True' == config.get('Data',
                                                       'write_sample_files')
        self.train_rows = config.getint('Data', 'train_rows')
        self.test_rows = config.getint('Data', 'test_rows')
        self.remove_after_run = config.getboolean('Data', 'remove_after_run')

    def download_data(self):

        os.chdir(self.project_path)

        if all(os.path.isfile(self.data_path + csv) for csv in self.csvs):
            print('Data already loaded')
        else:
            # if files not there yet init download it and write it as zip
            if not os.path.isfile('Expedia.zip'):
                print('Downloading Expedia.zip from Dropbox...')
                u = urllib.request.urlopen(self.url)
                data = u.read()
                u.close()

                with open('Expedia.zip', 'wb') as f:
                    f.write(data)
                print('Finished downloading Expedia.zip from Dropbox')

            # Extract gz files from zip
            with zipfile.ZipFile("Expedia.zip", 'r') as zip_ref:
                print('Unzipping Expedia.zip')
                zip_ref.extractall(self.project_path)
                print('Finished unzipping Expedia.zip...')

            os.chdir(self.project_path + '/all')

            # extract the csvs from the gzs
            for csv, gz in zip(self.csvs, self.gzs):
                with gzip.open(gz, 'rb') as f_in:
                    with open(csv, 'wb') as f_out:
                        print(f'writing {csv}...')
                        shutil.copyfileobj(f_in, f_out)
                        print(f'Finished writing {csv}')

            # clean up directory
            for f in self.gzs:
                os.remove(f)

            os.chdir(self.project_path)  # return to parent directory
            # if the data directory already exists probably empty, remove it
            if os.path.exists(self.data_path):  #
                shutil.rmtree(self.data_path)
            os.rename(self.project_path + '/all', self.data_path)
            os.remove('Expedia.zip')

            print(f'Finished.')

    def load_data(self):
        print('Loading data into memory / dataframe')
        if not self.is_development:
            df_train = pd.read_csv(self.data_path + self.csvs[0])
            df_test = pd.read_csv(self.data_path + self.csvs[1])
            # not even needed. its enough to have the destination id as feature
            df_destination = pd.read_csv(self.data_path + self.csvs[2])
        else:
            df_train = pd.read_csv(self.data_path + self.csvs[0],
                                   nrows=self.train_rows)
            df_test = pd.read_csv(self.data_path + self.csvs[1],
                                  nrows=self.test_rows)
            df_destination = pd.read_csv(self.data_path + self.csvs[2])

            if self.write_sample_files:
                print('Writing sample files...')
                df_train.to_csv(self.data_path + 'sample' + self.csvs[0],
                                index=False)
                df_test.to_csv(self.data_path + 'sample' + self.csvs[1],
                               index=False)
                print('Finished writing sample files...')

        print('Finished loading data into memory / dataframe')
        self.df_train = df_train
        self.df_test = df_test
        self.df_destination = df_destination

        if self.remove_after_run:
            self.remove_data()

    def remove_data(self):
        print('Removing files from hard drive...')
        cwd = os.getcwd()
        os.chdir(self.data_path)
        for csv in self.csvs:
            try:
                print('Removing ', csv, '...')
                os.remove(csv)
            except EnvironmentError:
                print(csv, ' not found.')
            try:
                print('Removing sample ', csv, '...')
                os.remove('sample' + csv)
            except EnvironmentError:
                print('sample', csv, ' not found.')
        os.chdir(cwd)
        print('Finished removing files from hard drive...')
