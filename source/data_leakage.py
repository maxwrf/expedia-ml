import pandas as pd
import numpy as np


class Exploit():
    def __init__(self, X_train, y_train, columns):
        self.columns = columns
        X_train = pd.DataFrame(X_train, columns=self.columns)
        y_train = pd.DataFrame(y_train, columns=['hotel_cluster'])
        self.exploit = pd.concat([X_train, y_train.reset_index().drop(['index'], axis=1)], axis=1)
        self.exploit = self.exploit[[
            'orig_destination_distance', 'user_location_city',
            'srch_destination_id', 'hotel_cluster', 'hotel_market']]
        self.exploit.drop_duplicates(inplace=True)

    def predict(self, X):
        X = pd.DataFrame(X, columns=self.columns)

        def lookup(instance):
            instance_orig_destination_distance = instance[
                'orig_destination_distance']
            instance_user_location_city = instance['user_location_city']
            instance_srch_destination_id = instance['srch_destination_id']
            instance_hotel_market = instance['hotel_market']
            pred = self.exploit[(self.exploit['orig_destination_distance']
                                 == instance_orig_destination_distance) &
                                (self.exploit['user_location_city']
                                    == instance_user_location_city) &
                                (self.exploit['srch_destination_id']
                                    == instance_srch_destination_id) &
                                (self.exploit['hotel_market']
                                    == instance_hotel_market)]

            if len(pred) != 1:
                # TODO: Why do I sometimes get multiple matches len > 1 ?
                return np.nan
            else:
                return int(pred['hotel_cluster'])
        preds = X.apply(lambda instance: lookup(instance), axis=1)
        return preds
