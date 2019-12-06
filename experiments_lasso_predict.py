from pp import perturbations
import numpy as np
from pp.meta_regressors import train_random_forest_regressor, evaluate_regressor
from pp.serialization import load_black_box
from pp.datasets import DATASETS_CATEGORICAL_COLUMNS


def missing_perturbations(substitute):
    _perturbations = []
    for fraction_of_values_to_delete in [0.0, 0.05, 0.25, 0.5, 0.75, 0.99]:
        for _ in range(0, 100):
            _perturbations.append(perturbations.MissingValues(fraction_of_values_to_delete, ['workclass'], substitute))

    return _perturbations


def custom_perturbations():
    _perturbations = []
    for _ in range(0, 10):
        _perturbations.append(perturbations.LassoExperiment('workclass', ['Local-gov', 'Never-worked', 'Without-pay']))
    return _perturbations

#workclass Local-gov is ignored by classifier
#workclass Never-worked is ignored by classifier
#workclass Without-pay is ignored by classifier
#99 1568 in test data
#109 1569 target data


models_to_evaluate = []
for learner in ['lasso']:
    for dataset in ['adult_minimal']:
        for score in ['accuracy']:
            models_to_evaluate.append(learner + '-' + dataset + '-' + score)

for model_to_evaluate in models_to_evaluate:
    experiment_name = 'correct_shift'

    (model, scoring, scoring_name, train_data, y_train, test_data, y_test, target_data, y_target, learner_name,
     dataset_name) = load_black_box(model_to_evaluate)

    performance_predictor = train_random_forest_regressor(test_data, y_test, missing_perturbations('NULL'),
                                                          model, scoring)

    evaluate_regressor(target_data, y_target, custom_perturbations(), model, performance_predictor, scoring,
                       scoring_name, dataset_name, 'missing', learner_name, experiment_name)
