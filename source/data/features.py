import numpy as np
from multiprocessing import Pool
import pandas as pd


class Features():
    def __init__(self, config, df_train, df_test):
        self.save_prepared = config.get('Features', 'save_prepared')
        self.df_train = df_train
        self.df_test = df_test
        self.project_path = config.get('General', 'path')
        self.data_path = self.project_path + '/data/'

    @staticmethod
    def fillna_convert(df):
        """Function to fill NANS and change time stamp foremats"""
        # filling empty orig_destianation distance
        df['orig_destination_distance'].fillna(
            df['orig_destination_distance'].median(), inplace=True)

        # make timestemps pd datetimes
        for col in df[['date_time', 'srch_ci', 'srch_co']].columns:
            df[col] = pd.to_datetime(df[col])

        return df

    @staticmethod
    def engineer_features(df, df_full):
        """
        Function to engineer features on the given df, aggregates user info
        from df_full
        """
        # length of trip
        df['trip_length'] = (df['srch_co'] - df['srch_ci'])\
            .astype('timedelta64[D]')
        df['trip_length'].fillna((df['trip_length'].median()), inplace=True)

        # is the trip in the same country
        df['domestic'] = np.where((df['user_location_country']
                                   .equals(df['hotel_country'])), 1, 0)

        # is it trip length smaller than 3 days
        df['short_trip'] = np.where((df['trip_length'] <= 3), 1, 0)

        #  is it a weekend trip
        df['srch_ci_d'] = df['srch_ci'].dt.day_name()
        df['srch_co_d'] = df['srch_co'].dt.day_name()
        df['weekend_trip'] = np.where((((df['srch_ci_d'] == 'Friday') &
                                        (df['trip_length'] <= 3)) |
                                       ((df['srch_ci_d'] == 'Saturday') &
                                        (df['trip_length'] <= 2))), 1, 0)

        # is it a business trip
        df['business_trip'] = np.where(((df['srch_ci_d'] != 'Friday') &
                                        (df['srch_ci_d'] != 'Saturday') &
                                        (df['srch_ci_d'] != 'Sunday') &
                                        (df['srch_co_d'] != 'Saturday') &
                                        (df['srch_co_d'] != 'Sunday') &
                                        (df['trip_length'] <= 4)), 1, 0)
        df.drop(columns=['srch_ci_d', 'srch_co_d'], inplace=True)

        # plan time - how far ahead do we plan the trip
        df['plan_time'] = (df['srch_ci'] - df['date_time'])\
            .astype('timedelta64[D]')

        # is it a solo trip / family trip
        df['solo_trip'] = np.where(((df['srch_adults_cnt'] == 1) &
                                    (df['srch_children_cnt'] == 0)), 1, 0)

        # aggregate a mean booking rate
        def aggregated_booking_rate(instance):
            if instance['is_booking'] == 0:
                return np.nan
            instance_date = instance['date_time']
            instance_id = instance['user_id']
            mean_booking_rate = df_full[(df_full['date_time'] <= instance_date)
                                        & (df_full['user_id'] == instance_id)
                                        ]['is_booking'].mean()
            return mean_booking_rate

        df['booking_rate'] = df.apply(aggregated_booking_rate, axis=1)

        # aggregate previous bookings & clicks by hotel cluster
        def aggregated_previous_cluster(instance, hotel_cluster):
            if instance['is_booking'] == 0:
                return np.nan, np.nan
            instance_date = instance['date_time']
            instance_id = instance['user_id']
            cnt_b = len(df_full[(df_full['date_time'] <= instance_date) &
                                (df_full['user_id'] == instance_id) &
                                (df_full['hotel_cluster'] == hotel_cluster) &
                                (df_full['is_booking'] == 1)])
            cnt_nob = len(df_full[(df_full['date_time'] <= instance_date) &
                                  (df_full['user_id'] == instance_id) &
                                  (df_full['hotel_cluster'] == hotel_cluster) &
                                  (df_full['is_booking'] == 0)])
            return cnt_b, cnt_nob

        for hotel_cluster in df_full['hotel_cluster'].unique():
            if np.isnan(hotel_cluster):  # test set does not have cluster given
                continue
            df['booked_cluster' + str(int(hotel_cluster))], \
                df['not_booked_cluster' + str(int(hotel_cluster))] = zip(
                *df.apply(lambda instance:
                          aggregated_previous_cluster(instance, hotel_cluster),
                          axis=1))

        return df

    @staticmethod
    def parallel_feature_engineering(df_to_split, df_full, n_cores=4):
        """Function to enable multi threading for feature engineering"""
        func = Features.engineer_features
        df_splits = np.array_split(df_to_split, n_cores)
        args = [[df_split, df_full] for df_split in df_splits]
        pool = Pool(n_cores)
        df = pd.concat(pool.starmap(func, args))
        pool.close()
        pool.join()
        return df

    @staticmethod
    def finalize(df):
        """Function removing features not needed for training"""
        for col in ['date_time', 'srch_ci', 'srch_co', 'user_id']:
            try:
                df.drop(col, axis=1, inplace=True)
            except KeyError:
                pass
        return df

    def prepare_df_train(self):
        """Function to build features for train df"""
        self.df_train = Features.fillna_convert(self.df_train)
        self.df_train = Features.parallel_feature_engineering(self.df_train,
                                                              self.df_train)
        self.df_train = Features.finalize(self.df_train)

        # only use rows where booking is 1 and aggregate all info
        len_before = len(self.df_train)
        self.df_train = self.df_train[self.df_train['is_booking'] == 1]
        self.df_train = self.df_train.drop('is_booking', axis=1)
        print(f'Dropped {len_before - len(self.df_train)} rows which did not\
                represent a booking in df train.')

        if self.save_prepared:
            self.df_train.to_csv(self.data_path + 'prepared_train.csv')

    def prepare_df_test(self):
        """Function to build features for train df"""
        self.df_test = Features.fillna_convert(self.df_test)
        df_train = Features.fillna_convert(self.df_train)

        # to engineer features on the test set we will concate test and train
        # and aggregate user info from train on test
        df_train['is_Train'] = True
        self.df_test['is_Train'] = False
        self.df_test['is_booking'] = 1
        self.df_test['hotel_cluster'] = np.nan
        df_test_full_hist = df_train.append(self.df_test)
        self.df_test = Features.parallel_feature_engineering(self.df_test,
                                                             df_test_full_hist)
        self.df_test = self.df_test.drop(['is_booking', 'is_Train',
                                          'hotel_cluster', 'id'], axis=1)
        self.df_test = Features.finalize(self.df_test)

        if self.save_prepared:
            self.df_test.to_csv(self.data_path + 'prepared_test.csv')
