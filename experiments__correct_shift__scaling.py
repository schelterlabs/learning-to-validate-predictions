from pp import perturbations
import numpy as np
from pp.meta_regressors import train_random_forest_regressor, evaluate_regressor
from pp.serialization import load_black_box
from pp.datasets import DATASETS_NUMERICAL_COLUMNS


def scaling_perturbations(_numerical_columns):
    _perturbations = []
    for num_columns_affected in range(1, len(_numerical_columns)):
        for fraction_of_outliers in [0.0, 0.05, 0.25, 0.5, 0.75, 0.99]:
            for _ in range(0, 100):
                columns_affected = np.random.choice(_numerical_columns, num_columns_affected)
                _perturbations.append(perturbations.Scaling(fraction_of_outliers, columns_affected))

    return _perturbations


models_to_evaluate = []
for learner in ['lr', 'xgb', 'dnn']:
    for dataset in ['adult', 'heart', 'bank']:
        for score in ['roc_auc']:
            models_to_evaluate.append(learner + '-' + dataset + '-' + score)

for model_to_evaluate in models_to_evaluate:
    experiment_name = 'correct_shift'

    (model, scoring, scoring_name, train_data, y_train, test_data, y_test, target_data, y_target, learner_name,
     dataset_name) = load_black_box(model_to_evaluate)

    numerical_columns = DATASETS_NUMERICAL_COLUMNS[dataset_name]

    performance_predictor = train_random_forest_regressor(test_data, y_test, scaling_perturbations(numerical_columns),
                                                          model, scoring)

    evaluate_regressor(target_data, y_target, scaling_perturbations(numerical_columns), model,
                       performance_predictor, scoring, scoring_name, dataset_name, 'scaling', learner_name,
                       experiment_name)
