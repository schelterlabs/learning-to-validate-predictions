# pickup/drop off location wrong due to gps issues (e.g. Brooklyn/Manhattan) -> tunnel
# total_amount reduced if RateCodeId = 5 (negotiated fare) &  payment type cash (not in credit)
#
# https://www1.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_green.pdf
# https://www1.nyc.gov/assets/tlc/images/content/pages/about/taxi_zone_map_brooklyn.jpg
# https://www1.nyc.gov/assets/tlc/images/content/pages/about/taxi_zone_map_manhattan.jpg
#
# https://eng.uber.com/rethinking-gps/
#
# MSE 0.00013, MAE 0.0113
#

from pp import perturbations
from pp.meta_regressors import train_random_forest_regressor, evaluate_regressor
from pp.serialization import load_black_box
import random
import numpy as np


class GPSProblemsPickup:

    adjacent_zones_brooklyn_manhattan = {
        '33': '87',
        '66': '209',
        '34': '45',
        '255': '232',
        '256': '232',
        '112': '4'
    }

    def __init__(self, probability):
        self.probability = probability

    def transform(self, clean_df):
        df = clean_df.copy(deep=True)

        for index, row in df.iterrows():
            # Taxi lost connection to server and dropped passenger off in a zone adjacent to manhattan
            if row['store_and_fwd_flag'] == 'Y' and row['PULocationID'] in self.adjacent_zones_brooklyn_manhattan\
                    and random.random() < self.probability:
                # Adjacent zone from manhattan is reported
                manhattan_zone_id = self.adjacent_zones_brooklyn_manhattan[row['PULocationID']]
                df.at[index, 'PULocationID'] = manhattan_zone_id
        return df


class GPSProblemsDropOff:

    adjacent_zones_brooklyn_manhattan = {
        '33': '87',
        '66': '209',
        '34': '45',
        '255': '232',
        '256': '232',
        '112': '4'
    }

    def __init__(self, probability):
        self.probability = probability

    def transform(self, clean_df):
        df = clean_df.copy(deep=True)

        for index, row in df.iterrows():
            # Taxi lost connection to server and dropped passenger off in a zone adjacent to manhattan
            #if row['store_and_fwd_flag'] == 'Y' and row['DOLocationID'] in self.adjacent_zones_brooklyn_manhattan\
            if row['DOLocationID'] in self.adjacent_zones_brooklyn_manhattan \
                    and random.random() < self.probability:
                # Adjacent zone from manhattan is reported
                manhattan_zone_id = self.adjacent_zones_brooklyn_manhattan[row['DOLocationID']]
                df.at[index, 'DOLocationID'] = manhattan_zone_id
        return df


class TotalAmountReduced:

    def __init__(self, probability):
        self.probability = probability

    def transform(self, clean_df):
        df = clean_df.copy(deep=True)

        for index, row in df.iterrows():
            # Negotiated fare, paid in cash
            if row['RatecodeID'] == '5' and row['payment_type'] == '2' and random.random() < self.probability:
                # Assume driver does not report full amount
                reduced_amount = row['total_amount'] * np.random.uniform(0.5, 0.75)
                df.at[index, 'total_amount'] = reduced_amount
        return df



def domain_perturbations():
    _perturbations = []
    for _ in range(0, 500):
            _perturbations.append(perturbations.Mixture([
                GPSProblemsPickup(random.random()),
                GPSProblemsDropOff(random.random()),
                TotalAmountReduced(random.random()),
            ]))

    return _perturbations


models_to_evaluate = []
for learner in ['autosklearn']:
    for dataset in ['taxi']:
        for score in ['accuracy']:
            models_to_evaluate.append(learner + '-' + dataset + '-' + score)

for model_to_evaluate in models_to_evaluate:
    experiment_name = 'domain-experts'

    (model, scoring, scoring_name, train_data, y_train, test_data, y_test, target_data, y_target, learner_name,
     dataset_name) = load_black_box(model_to_evaluate)

    predictor = train_random_forest_regressor(test_data, y_test, domain_perturbations(), model, scoring)

    evaluate_regressor(target_data, y_target, domain_perturbations(), model,
                       predictor, scoring, scoring_name, dataset_name, 'noisy', learner_name, experiment_name)
