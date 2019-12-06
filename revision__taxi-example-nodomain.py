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
import itertools
from pp import perturbations
from pp.meta_regressors import train_random_forest_regressor, evaluate_regressor
from pp.serialization import load_black_box
import random
from pp.datasets import DATASETS_CATEGORICAL_COLUMNS, DATASETS_NUMERICAL_COLUMNS
from pp.generators import *
import numpy as np
from pp.perturbations import Mixture

num_repetitions = 50


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
    experiment_name = 'no-domain-expertise'

    (model, scoring, scoring_name, train_data, y_train, test_data, y_test, target_data, y_target, learner_name,
     dataset_name) = load_black_box(model_to_evaluate)

    categorical_columns = DATASETS_CATEGORICAL_COLUMNS[dataset_name]
    numerical_columns = DATASETS_NUMERICAL_COLUMNS[dataset_name]

    swap_affected_column_pairs = list(itertools.chain(
        itertools.combinations(numerical_columns, 2), itertools.combinations(categorical_columns, 2)))
    affected_column_pairs = random.sample(swap_affected_column_pairs, 5)

    probs = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.99]

    available_perturbations = {
        'swapped': swapped_perturbations(swap_affected_column_pairs, num_repetitions, probs),
        'missing': missing_perturbations(categorical_columns, 'NULL', num_repetitions, probs),
        'outlier': outlier_perturbations(numerical_columns, num_repetitions, probs),
        'scaling': scaling_perturbations(numerical_columns, num_repetitions, probs),
    }

    random.shuffle(available_perturbations['swapped'])
    random.shuffle(available_perturbations['missing'])
    random.shuffle(available_perturbations['outlier'])
    random.shuffle(available_perturbations['scaling'])

    train_perturbations = []
    test_perturbations = []
    for a, b, c, d in zip(available_perturbations['swapped'],
                          available_perturbations['missing'],
                          available_perturbations['outlier'],
                          available_perturbations['scaling']):
        train_perturbations.append(Mixture([a, b, c, d]))

    random.shuffle(available_perturbations['swapped'])
    random.shuffle(available_perturbations['missing'])
    random.shuffle(available_perturbations['outlier'])
    random.shuffle(available_perturbations['scaling'])

    for a, b, c, d in zip(available_perturbations['swapped'],
                          available_perturbations['missing'],
                          available_perturbations['outlier'],
                          available_perturbations['scaling']):
        test_perturbations.append(Mixture([a, b, c, d]))


    predictor = train_random_forest_regressor(test_data, y_test, train_perturbations, model, scoring)

    evaluate_regressor(target_data, y_target, domain_perturbations(), model,
                       predictor, scoring, scoring_name, dataset_name, 'noisy', learner_name, experiment_name)
