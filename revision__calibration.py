from pp.meta_regressors import train_random_forest_regressor, evaluate_regressor_against_corrupted_calibration_baseline
from pp.serialization import load_black_box
from pp.datasets import DATASETS_CATEGORICAL_COLUMNS, DATASETS_NUMERICAL_COLUMNS
from pp.generators import missing_perturbations, outlier_perturbations
from pp.perturbations import Mixture
from pp.learners import LogisticRegression,XgBoost,DNN
from pp.datasets import BalancedAdultDataset

import random

num_repetitions = 100

models_to_evaluate = []
for learner in ['dnn']:
    for dataset in ['adult']:
        for score in ['accuracy']:

            model_to_evaluate = learner + '-' + dataset + '-' + score

            experiment_name = 'revision__calibration_correct_shift__missing'

            (model, scoring, scoring_name, train_data, y_train, test_data, y_test, target_data, y_target, learner_name,
             dataset_name) = load_black_box(model_to_evaluate)

            # TODO ADJUST MANUALLY
            #learner_to_fit = LogisticRegression('accuracy')
            #learner_to_fit = XgBoost('accuracy')
            learner_to_fit = DNN('accuracy')
            dataset_to_use = BalancedAdultDataset()

            categorical_columns = DATASETS_CATEGORICAL_COLUMNS[dataset_name]
            numerical_columns = DATASETS_NUMERICAL_COLUMNS[dataset_name]

            probs = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.99]

            available_perturbations = {
                'missing': missing_perturbations(categorical_columns, 'NULL', num_repetitions, probs),
                'outlier': outlier_perturbations(numerical_columns, num_repetitions, probs)
            }

            random.shuffle(available_perturbations['missing'])
            random.shuffle(available_perturbations['outlier'])

            train_perturbations = []
            for a, b in zip(available_perturbations['missing'], available_perturbations['outlier']):
                train_perturbations.append(Mixture([a, b]))

            random.shuffle(available_perturbations['missing'])
            random.shuffle(available_perturbations['outlier'])

            test_perturbations = []
            for a, b in zip(available_perturbations['missing'], available_perturbations['outlier']):
                test_perturbations.append(Mixture([a, b]))

            performance_predictor = train_random_forest_regressor(test_data, y_test, train_perturbations, model, scoring)

            evaluate_regressor_against_corrupted_calibration_baseline(test_data, y_test, target_data, y_target,
                                                                      test_perturbations, model, performance_predictor,
                                                                      scoring, scoring_name, dataset_name,
                                                                      'missing', learner_name, experiment_name,
                                                                      'sigmoid', learner_to_fit, dataset_to_use)

