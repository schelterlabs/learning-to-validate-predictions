from pp import perturbations
from pp.meta_regressors import train_random_forest_regressor, evaluate_regressor
from pp.serialization import load_black_box


def rotation_perturbations():
    _perturbations = []
    for fraction in [0.0, 0.05, 0.25, 0.5, 0.75, 0.99]:
        for _ in range(0, 100):
            _perturbations.append(perturbations.ImageRotation(fraction))

    return _perturbations


models_to_evaluate = []
for learner in ['convnet']:
    for dataset in ['mnist', 'fashion']:
        for score in ['roc_auc']:
            models_to_evaluate.append(learner + '-' + dataset + '-' + score)

for model_to_evaluate in models_to_evaluate:
    experiment_name = 'correct_shift'

    (model, scoring, scoring_name, train_data, y_train, test_data, y_test, target_data, y_target, learner_name,
     dataset_name) = load_black_box(model_to_evaluate)

    predictor = train_random_forest_regressor(test_data, y_test, rotation_perturbations(), model, scoring)

    evaluate_regressor(target_data, y_target, rotation_perturbations(), model,
                       predictor, scoring, scoring_name, dataset_name, 'rotation', learner_name, experiment_name)
