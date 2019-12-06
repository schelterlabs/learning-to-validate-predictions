from pp.meta_regressors import train_random_forest_regressor_with_noise, evaluate_regressor_with_noise
from pp.serialization import load_black_box
from pp.datasets import DATASETS_NUMERICAL_COLUMNS, DATASETS_CATEGORICAL_COLUMNS
from pp.generators import outlier_perturbations, missing_perturbations

import random
import numpy as np

num_repetitions = 250

models_to_evaluate = []
for learner in ['lr']:
    for dataset in ['adult']:
        for score in ['accuracy']:
            models_to_evaluate.append(learner + '-' + dataset + '-' + score)

for model_to_evaluate in models_to_evaluate:

    experiment_name = 'noise_on_features__missing__noiseeval'

    (model, scoring, scoring_name, train_data, y_train, test_data, y_test, target_data, y_target, learner_name,
     dataset_name) = load_black_box(model_to_evaluate)

    categorical_columns = DATASETS_CATEGORICAL_COLUMNS[dataset_name]

    performance_predictor = train_random_forest_regressor_with_noise(test_data, y_test, num_repetitions,
                                                                     model, scoring)

    evaluate_regressor_with_noise(target_data, y_target, num_repetitions,
                       model, performance_predictor, scoring, scoring_name, dataset_name, 'missing', learner_name,
                       experiment_name)