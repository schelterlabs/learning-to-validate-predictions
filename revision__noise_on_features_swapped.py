from pp.meta_regressors import train_random_forest_regressor_with_noise, evaluate_regressor
from pp.serialization import load_black_box
from pp.datasets import DATASETS_NUMERICAL_COLUMNS, DATASETS_CATEGORICAL_COLUMNS
from pp.generators import outlier_perturbations, missing_perturbations
from pp.perturbations import SwappedValues
import itertools
import random

def swapped_perturbations(_affected_column_pairs):
    _perturbations = []
    for column_pair in _affected_column_pairs:
        for fraction_of_outliers in [0.0, 0.05, 0.25, 0.5, 0.75, 0.99]:
            for _ in range(0, 100):
                _perturbations.append(SwappedValues(fraction_of_outliers, column_pair))

    return _perturbations


num_repetitions = 100

models_to_evaluate = []
for learner in ['lr']:
    for dataset in ['adult']:
        for score in ['accuracy']:
            models_to_evaluate.append(learner + '-' + dataset + '-' + score)

for model_to_evaluate in models_to_evaluate:

    experiment_name = 'noise_on_features__swapped'

    (model, scoring, scoring_name, train_data, y_train, test_data, y_test, target_data, y_target, learner_name,
     dataset_name) = load_black_box(model_to_evaluate)

    categorical_columns = DATASETS_CATEGORICAL_COLUMNS[dataset_name]
    numerical_columns = DATASETS_NUMERICAL_COLUMNS[dataset_name]

    affected_column_pairs = list(itertools.chain(
        itertools.combinations(numerical_columns, 2), itertools.combinations(categorical_columns, 2)))

    affected_column_pairs = random.sample(affected_column_pairs, 5)
    performance_predictor = train_random_forest_regressor_with_noise(test_data, y_test, num_repetitions,
                                                                     model, scoring)

    evaluate_regressor(target_data, y_target, swapped_perturbations(affected_column_pairs),
                       model, performance_predictor, scoring, scoring_name, dataset_name, 'swapped', learner_name,
                       experiment_name)
