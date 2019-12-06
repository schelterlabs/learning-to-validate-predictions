
from pp.meta_regressors import train_random_forest_regressor, evaluate_regressor
from pp.serialization import load_black_box
from pp.datasets import DATASETS_NUMERICAL_COLUMNS, DATASETS_CATEGORICAL_COLUMNS
from pp.generators import outlier_perturbations, missing_perturbations

import random

num_repetitions = 100

models_to_evaluate = []
for learner in ['dnn']:
    for dataset in ['heart']:
        for score in ['accuracy']:
            models_to_evaluate.append(learner + '-' + dataset + '-' + score)

for model_to_evaluate in models_to_evaluate:

    for sample_size in [10, 50, 100, 250, 500, 750, 1000, 1500]:

        experiment_name = 'dtest_variation__' + str(sample_size)

        (model, scoring, scoring_name, train_data, y_train, test_data, y_test, target_data, y_target, learner_name,
         dataset_name) = load_black_box(model_to_evaluate)

        categorical_columns = DATASETS_CATEGORICAL_COLUMNS[dataset_name]
        numerical_columns = DATASETS_NUMERICAL_COLUMNS[dataset_name]

        # Downsample test_data and y_test
        indices = list(range(0, len(test_data)))
        random.shuffle(indices)
        num_samples_to_take = sample_size
        indices_to_take = indices[:num_samples_to_take]

        test_data_sampled = test_data.iloc[indices_to_take, :]
        y_test_sampled = []
        for index in indices_to_take:
            y_test_sampled.append(y_test[index])



        performance_predictor = train_random_forest_regressor(test_data_sampled, y_test_sampled,
                                                              outlier_perturbations(numerical_columns, num_repetitions),
                                                              model, scoring)

        evaluate_regressor(target_data, y_target, outlier_perturbations(numerical_columns, num_repetitions),
                           model, performance_predictor, scoring, scoring_name, dataset_name, 'missing', learner_name,
                           experiment_name)

        # performance_predictor = train_random_forest_regressor(test_data_sampled, y_test_sampled,
        #                                                       missing_perturbations(categorical_columns,
        #                                                                                                'NULL',
        #                                                                                                num_repetitions),
        #                                                       model, scoring)
        #
        # evaluate_regressor(target_data, y_target, missing_perturbations(categorical_columns, 'NULL', num_repetitions),
        #                    model, performance_predictor, scoring, scoring_name, dataset_name, 'missing', learner_name,
        #                    experiment_name)
