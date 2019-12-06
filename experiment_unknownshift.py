from pp.meta_classifier import train_performance_predictor, evaluate_predictor
from pp.serialization import load_black_box
from pp.datasets import DATASETS_CATEGORICAL_COLUMNS, DATASETS_NUMERICAL_COLUMNS
from pp.generators import *
import itertools
import random
from pp.perturbations import Mixture

num_repetitions = 100


for threshold in [0.1]:
    models_to_evaluate = []
    for dataset in ['heart']:
        for learner in ['xgb']:
            for score in ['accuracy']:
                models_to_evaluate.append(learner + '-' + dataset + '-' + score)


    for model_to_evaluate in models_to_evaluate:
        experiment_name = 'wild_mixture'

        (model, scoring, scoring_name, train_data, y_train, test_data, y_test, target_data, y_target, learner_name,
         dataset_name) = load_black_box(model_to_evaluate)

        categorical_columns = DATASETS_CATEGORICAL_COLUMNS[dataset_name]
        numerical_columns = DATASETS_NUMERICAL_COLUMNS[dataset_name]

        swap_affected_column_pairs = list(itertools.chain(
            itertools.combinations(numerical_columns, 2), itertools.combinations(categorical_columns, 2)))
        affected_column_pairs = random.sample(swap_affected_column_pairs, 5)

        probs = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.99]

        available_perturbations = {
            'swapped': swapped_perturbations(swap_affected_column_pairs, num_repetitions, probs),
            'missing': missing_perturbations(categorical_columns, 'NULL', num_repetitions, probs),
            'outlier': outlier_perturbations(numerical_columns, num_repetitions, probs),
            'scaling': scaling_perturbations(numerical_columns, num_repetitions, probs),
        }

        random.shuffle(available_perturbations['swapped'])
        random.shuffle(available_perturbations['missing'])
        random.shuffle(available_perturbations['outlier'])
        random.shuffle(available_perturbations['scaling'])

        train_perturbations = []
        for a, b, c, d in zip(available_perturbations['swapped'],
                              available_perturbations['missing'],
                              available_perturbations['outlier'],
                              available_perturbations['scaling']):
            train_perturbations.append(Mixture([a, b, c, d]))

        unknown_perturbations = {
            'plusminus': plusminus_perturbations(numerical_columns, num_repetitions, probs),
            'typos': typo_perturbations(categorical_columns, num_repetitions, probs),
            'flipsigns': flipsign_perturbations(categorical_columns, num_repetitions, probs),
        }

        random.shuffle(unknown_perturbations['plusminus'])
        random.shuffle(unknown_perturbations['typos'])
        random.shuffle(unknown_perturbations['flipsigns'])

        test_perturbations = []
        for a, b, c in zip(unknown_perturbations['plusminus'], unknown_perturbations['typos'], unknown_perturbations['flipsigns']):
            test_perturbations.append(Mixture([a, b, c]))


        performance_predictor = train_performance_predictor(test_data, y_test, threshold, train_perturbations, model,
                                                            scoring)

        evaluate_predictor(target_data, y_target, threshold, test_perturbations, model, performance_predictor, scoring,
                           scoring_name, dataset_name, 'mixture_with_unknown', learner_name, experiment_name,
                           test_data, y_test, categorical_columns, numerical_columns)
