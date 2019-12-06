from pp import perturbations
from pp.meta_regressors import train_random_forest_regressor, evaluate_regressor
from pp.serialization import load_black_box
from pp.datasets import DATASETS_NUMERICAL_COLUMNS, DATASETS_CATEGORICAL_COLUMNS
import itertools
import random


def swapped_perturbations(_affected_column_pairs):
    _perturbations = []
    for column_pair in _affected_column_pairs:
        for fraction_of_outliers in [0.0, 0.05, 0.25, 0.5, 0.75, 0.99]:
            for _ in range(0, 100):
                _perturbations.append(perturbations.SwappedValues(fraction_of_outliers, column_pair))

    return _perturbations


models_to_evaluate = []
for learner in ['lr', 'dnn', 'xgb']:
    for dataset in ['adult', 'heart', 'bank']:
        for score in ['roc_auc']:
            models_to_evaluate.append(learner + '-' + dataset + '-' + score)

for model_to_evaluate in models_to_evaluate:
    experiment_name = 'correct_shift'

    (model, scoring, scoring_name, train_data, y_train, test_data, y_test, target_data, y_target, learner_name,
     dataset_name) = load_black_box(model_to_evaluate)

    categorical_columns = DATASETS_CATEGORICAL_COLUMNS[dataset_name]
    numerical_columns = DATASETS_NUMERICAL_COLUMNS[dataset_name]

    affected_column_pairs = list(itertools.chain(
        itertools.combinations(numerical_columns, 2), itertools.combinations(categorical_columns, 2)))

    affected_column_pairs = random.sample(affected_column_pairs, 5)

    predictor = train_random_forest_regressor(test_data, y_test, swapped_perturbations(affected_column_pairs),
                                              model, scoring)

    evaluate_regressor(target_data, y_target, swapped_perturbations(affected_column_pairs), model,
                       predictor, scoring, scoring_name, dataset_name, 'swapped', learner_name, experiment_name)
