import warnings
# needed to get rid of deprecation warnings
import imp
from joblib import dump
from pp import datasets
from pp import learners
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
from pp.meta_classifier import train_performance_predictor, evaluate_predictor
from pp.datasets import DATASETS_CATEGORICAL_COLUMNS, DATASETS_NUMERICAL_COLUMNS
from pp.generators import *
import itertools
import random
from pp.perturbations import Mixture
from sklearn.metrics import accuracy_score

# We have to train the model here as TPOT does not support pickling of its models
dataset = datasets.BalancedAdultDataset()
learner = learners.TPOT('accuracy')

train_data, test_data, target_data = learner.split(dataset.df)

y_train = dataset.labels_from(train_data)
y_test = dataset.labels_from(test_data)
y_target = dataset.labels_from(target_data)

model = learner.fit(dataset, train_data)

score_on_train_data = learner.score(y_train, model.predict(train_data))
score_on_noncorrupted_test_data = learner.score(y_test, model.predict(test_data))
score_on_noncorrupted_target_data = learner.score(y_target, model.predict(target_data))

print(learner.scoring, "on train data: ", score_on_train_data)
print(learner.scoring, "on test data: ", score_on_noncorrupted_test_data)
print(learner.scoring, "on target data: ", score_on_noncorrupted_target_data)

dataset_name = 'adult'
num_repetitions = 20


for threshold in [0.03, 0.05, 0.1]:

    for model_to_evaluate in [model]:
        experiment_name = 'wild_mixture'

        scoring = accuracy_score
        scoring_name = 'accuracy'
        learner_name = 'tpot'

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
        test_perturbations = []
        for a, b, c, d in zip(available_perturbations['swapped'],
                              available_perturbations['missing'],
                              available_perturbations['outlier'],
                              available_perturbations['scaling']):
            train_perturbations.append(Mixture([a, b, c, d]))

        random.shuffle(available_perturbations['swapped'])
        random.shuffle(available_perturbations['missing'])
        random.shuffle(available_perturbations['outlier'])
        random.shuffle(available_perturbations['scaling'])

        for a, b, c, d in zip(available_perturbations['swapped'],
                              available_perturbations['missing'],
                              available_perturbations['outlier'],
                              available_perturbations['scaling']):
            test_perturbations.append(Mixture([a, b, c, d]))


        performance_predictor = train_performance_predictor(test_data, y_test, threshold, train_perturbations, model,
                                                            scoring)

        evaluate_predictor(target_data, y_target, threshold, test_perturbations, model, performance_predictor, scoring,
                           scoring_name, dataset_name, 'mixture', learner_name, experiment_name,
                           test_data, y_test, categorical_columns, numerical_columns)
