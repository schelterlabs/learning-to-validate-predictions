from pp import perturbations
from pp.meta_regressors import train_random_forest_regressor, evaluate_regressor
from pp.serialization import load_black_box


def leet_perturbations():
    _perturbations = []
    for fraction_of_troll_tweets in [0.0, 0.05, 0.25, 0.5, 0.75, 0.99]:
        for _ in range(0, 100):
            _perturbations.append(perturbations.Leetspeak(fraction_of_troll_tweets, 'content', 'label', 1))

    return _perturbations


models_to_evaluate = []
for learner in ['lr', 'dnn', 'xgb']:
    for dataset in ['trolling']:
        for score in ['roc_auc']:
            models_to_evaluate.append(learner + '-' + dataset + '-' + score)

for model_to_evaluate in models_to_evaluate:
    experiment_name = 'correct_shift'

    (model, scoring, scoring_name, train_data, y_train, test_data, y_test, target_data, y_target, learner_name,
     dataset_name) = load_black_box(model_to_evaluate)

    predictor = train_random_forest_regressor(test_data, y_test, leet_perturbations(), model, scoring)

    evaluate_regressor(target_data, y_target, leet_perturbations(), model,
                       predictor, scoring, scoring_name, dataset_name, 'adversarial', learner_name, experiment_name)
