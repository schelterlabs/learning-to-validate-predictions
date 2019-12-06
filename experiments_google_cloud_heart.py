"""
Hacky implementation of experiments where the production model is trained
in Google Cloud Auto ML and the performance predictor offline.
"""

from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from pp.meta_regressors import compute_features
from pp.datasets import DATASETS_CATEGORICAL_COLUMNS, DATASETS_NUMERICAL_COLUMNS
from pp.generators import *
import itertools
import random
from sklearn.pipeline import Pipeline
from pp.perturbations import Mixture
import pandas as pd
import os
# from pp.learners import Learner

def generate_training_data():

    data = pd.read_csv('./datasets/cardio/cardio_train.csv', delimiter=';')
    data[[col for col in data.columns if not col == 'id']]

    data['bmi'] = data['weight'] / (.01 * data['height']) ** 2
    data['age_in_years'] = data['age'] / 365.25

    # train_data, test_data, target_data = Learner.split(data)
    train_data, target_data = train_test_split(data, test_size=0.5)

    dataset_name = 'heart'
    num_repetitions = 20
    probs = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.99]

    categorical_columns = DATASETS_CATEGORICAL_COLUMNS[dataset_name]
    numerical_columns = DATASETS_NUMERICAL_COLUMNS[dataset_name]

    swap_affected_column_pairs = list(itertools.chain(
        itertools.combinations(numerical_columns, 2), itertools.combinations(categorical_columns, 2)))

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

    corrupted_test_data_list = []
    # train_perturbations = train_perturbations[:3]
    for count, perturbation in enumerate(train_perturbations):
        print(str(count) + '/' + str(len(train_perturbations)))
        corrupted_test_data = perturbation.transform(target_data)
        corrupted_test_data['perturbation_id'] = count
        corrupted_test_data_list.append(corrupted_test_data)

    if not os.path.exists("./perturbed_datasets_heart/"):
        os.mkdir('./perturbed_datasets_heart')

    pd.concat(corrupted_test_data_list).\
        to_csv('./perturbed_datasets_heart/heart_all_shifts_combined.csv')

    train_data.to_csv('./perturbed_datasets_heart/heart_train.csv')


def train_meta_regressor(predictions_perturbed, perturbation_ids_train, perturbation_ids_valid):

    scoring = accuracy_score
    num_examples = 0
    generated_training_data = []
    for perturbation_id in sorted(predictions_perturbed.perturbation_id.unique()):

        predictions = predictions_perturbed[predictions_perturbed.perturbation_id == perturbation_id]
        predictions_proba = np.array(predictions[['cardio_1_score', 'cardio_0_score']])
        features = compute_features(predictions_proba)
        num_features = features.shape[0]

        predictions['prediction'] = predictions['cardio_0_score'].apply(lambda row: 0 if row > .5 else 1)

        score_on_corrupted_test_data = scoring(predictions['cardio'], predictions['prediction'])

        example = np.concatenate((features, [score_on_corrupted_test_data]), axis=0)

        generated_training_data.append(example)

        num_examples += 1
        if num_examples % 10 == 0:
            print(num_examples)

    X = np.array(generated_training_data)[perturbation_ids_train, :num_features]
    y = np.array(generated_training_data)[perturbation_ids_train, num_features]

    X_valid = np.array(generated_training_data)[perturbation_ids_valid, :num_features]
    y_valid = np.array(generated_training_data)[perturbation_ids_valid, num_features]

    param_grid = {
        'learner__n_estimators': np.arange(5, 20, 5),
        'learner__criterion': ['mae']
    }

    meta_regressor_pipeline = Pipeline([
       ('scaling', StandardScaler()),
       ('learner', RandomForestRegressor(criterion='mae'))
    ])

    print(X.shape)

    print("Training performance predictor...")
    meta_regressor = GridSearchCV(meta_regressor_pipeline, param_grid, scoring='neg_mean_absolute_error').fit(X, y)

    print("Done with training")

    return meta_regressor, [X_valid, y_valid]


if __name__ == "__main__":

    # 1) generate training data
    # generate_training_data()

    # 2) Offline Go to any cloudservice you like, train a model and run predictions on the perturbed data

    # 3) Load predictions
    filenames_to_load = []
    for (dirpath, dirnames, filenames) in os.walk('./perturbed_datasets_heart/all_shifts_combined_predictions'):
        for filename in filenames:
            if 'table' in filename:
                filenames_to_load.append(filename)

    predictions_perturbed = pd.concat(
        [pd.read_csv(os.path.join('./perturbed_datasets_heart/all_shifts_combined_predictions', filename))
                     for filename in filenames_to_load])

    perturbations_ids = predictions_perturbed.perturbation_id.unique()
    perturbations_ids_valid = np.random.choice(perturbations_ids, 50)
    perturbations_ids_train = [pid for pid in perturbations_ids if pid not in perturbations_ids_valid]

    # 4) Train meta regressor
    meta_regressor, (X_valid, y_valid) = train_meta_regressor(predictions_perturbed,
                                                              perturbations_ids_train,
                                                              perturbations_ids_valid)

    # 5) Validate
    predictions_proba = meta_regressor.best_estimator_.predict(X_valid)

    # 6) Save results
    tmp = pd.DataFrame(np.array([y_valid, predictions_proba]).transpose(), columns=['true', 'predicted'])
    tmp.to_csv(
        '/Users/tammruka/Projects/projects-black-box-performance-prediction/artifacts/cloud_service/heart/results.csv')

    # 7) Plot results
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = sns.scatterplot(y_valid, predictions_proba, ax=ax)
    ax.set_xlabel('True model accuracy')
    ax.set_ylabel('Predicted model accuracy')
    ax.set_title('Cloud AutoML performance predictor on heart data')
    fig.savefig('./artifacts/cloud_service/validation_cardio.png')
    fig.show()