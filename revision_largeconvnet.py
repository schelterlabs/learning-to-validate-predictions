from pp import perturbations
from pp.meta_classifier import train_performance_predictor, evaluate_predictor
from pp.serialization import load_black_box
import random
from pp.perturbations import Mixture


def rotation_perturbations(probabilities, num_repetitions):
    _perturbations = []
    for fraction in probabilities:
        for _ in range(0, num_repetitions):
            _perturbations.append(perturbations.ImageRotation(fraction))

    return _perturbations


def noise_perturbations(probabilities, num_repetitions):
    _perturbations = []
    for fraction in probabilities:
        for _ in range(0, num_repetitions):
            _perturbations.append(perturbations.NoisyImage(fraction))

    return _perturbations


num_repetitions = 20
probs = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.99]

for threshold in [0.05, 0.1]:#[0.03, 0.05, 0.1]:
    models_to_evaluate = []
    for dataset in ['mnist']:
        for learner in ['autokeraslarge']:
            for score in ['accuracy']:
                models_to_evaluate.append(learner + '-' + dataset + '-' + score)


    for model_to_evaluate in models_to_evaluate:
        experiment_name = 'largeconvnet_wild_mixture'

        (model, scoring, scoring_name, train_data, y_train, test_data, y_test, target_data, y_target, learner_name,
         dataset_name) = load_black_box(model_to_evaluate)

        #print(model.clf)

        available_perturbations = {
            'rotations': rotation_perturbations(probs, num_repetitions),
            'noise': noise_perturbations(probs, num_repetitions)
        }

        random.shuffle(available_perturbations['rotations'])
        random.shuffle(available_perturbations['noise'])

        train_perturbations = []
        test_perturbations = []
        for a, b in zip(available_perturbations['rotations'], available_perturbations['noise']):
            train_perturbations.append(Mixture([a, b]))

        random.shuffle(available_perturbations['rotations'])
        random.shuffle(available_perturbations['noise'])


        for a, b in zip(available_perturbations['rotations'], available_perturbations['noise']):
            test_perturbations.append(Mixture([a, b]))


        performance_predictor = train_performance_predictor(test_data, y_test, threshold, train_perturbations, model,
                                                            scoring)

        evaluate_predictor(target_data, y_target, threshold, test_perturbations, model, performance_predictor, scoring,
                           scoring_name, dataset_name, 'largeconvnet_mixture', learner_name, experiment_name,
                           test_data, y_test, [], [])
