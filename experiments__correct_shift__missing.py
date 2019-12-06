
from pp.meta_regressors import train_random_forest_regressor, evaluate_regressor
from pp.serialization import load_black_box
from pp.datasets import DATASETS_CATEGORICAL_COLUMNS
from pp.generators import missing_perturbations

threshold = 0.05
num_repetitions = 100

models_to_evaluate = []
for learner in ['xgb']:
    for dataset in ['adult']:
        for score in ['accuracy']:
            models_to_evaluate.append(learner + '-' + dataset + '-' + score)

for model_to_evaluate in models_to_evaluate:
    experiment_name = 'playing_correct_shift'

    (model, scoring, scoring_name, train_data, y_train, test_data, y_test, target_data, y_target, learner_name,
     dataset_name) = load_black_box(model_to_evaluate)

    categorical_columns = DATASETS_CATEGORICAL_COLUMNS[dataset_name]

    performance_predictor = train_random_forest_regressor(test_data, y_test, missing_perturbations(categorical_columns,
                                                                                                   'NULL',
                                                                                                   num_repetitions),
                                                          model, scoring)

    evaluate_regressor(target_data, y_target, threshold, missing_perturbations(categorical_columns, 'NULL',
                                                                               num_repetitions),
                       model, performance_predictor, scoring, scoring_name, dataset_name, 'missing', learner_name)
