from pp.serialization import load_black_box
from pp.datasets import DATASETS_CATEGORICAL_COLUMNS, DATASETS_NUMERICAL_COLUMNS
from pp.generators import *
import random
from pp.meta_regressors import train_random_forest_regressor, evaluate_regressor_sliced
from pp.perturbations import Mixture

num_repetitions = 10
experiment_name = 'slices'


class OutliersInSlice:

    def __init__(self, row_predicate, columns, info):
        self.predicate = row_predicate
        self.columns = columns
        self.info = info

    def transform(self, clean_df):
        # we operate on a copy of the data
        df = clean_df.copy(deep=True)
        stddevs = {column: np.std(df[column]) for column in self.columns}
        scales = {column: random.uniform(1, 5) for column in self.columns}

        for index, row in df.iterrows():
            if self.predicate(row):
                for column in self.columns:
                    noise = np.random.normal(0, scales[column] * stddevs[column])
                    outlier = df.at[index, column] + noise

                    df.at[index, column] = outlier

        return df

    def name(self):
        return self.info

    def num_affected(self, df):
        return len(df[self.predicate(df)])


def add_perturbations(_perturbations, _predicate, info):
    for _num_columns_affected in range(1, len(categorical_columns) + 1):
        for _ in range(0, num_repetitions):
            _columns_affected = np.random.choice(categorical_columns, _num_columns_affected)

            _num_numerical_columns_affected = np.random.randint(len(numerical_columns)) + 1
            _numerical_columns_affected = np.random.choice(numerical_columns, _num_numerical_columns_affected)

            _perturbations.append(OutliersInSlice(_predicate, _numerical_columns_affected, info))




models_to_evaluate = []
for dataset in ['heart']:
    for learner in ['lr', 'xgb', 'dnn']:
        for score in ['accuracy']:
            models_to_evaluate.append(learner + '-' + dataset + '-' + score)

for model_to_evaluate in models_to_evaluate:

    (model, scoring, scoring_name, train_data, y_train, test_data, y_test, target_data, y_target, learner_name,
     dataset_name) = load_black_box(model_to_evaluate)

    perturbations = []
    categorical_columns = DATASETS_CATEGORICAL_COLUMNS[dataset_name]
    numerical_columns = DATASETS_NUMERICAL_COLUMNS[dataset_name]


    for gender in [1, 2]:
        predicate = lambda df: (df.gender == gender)
        add_perturbations(perturbations, predicate, "gender=="+str(gender))

    for alco in [0, 1]:
        predicate = lambda df: (df.alco == alco)
        add_perturbations(perturbations, predicate, "aloc=="+str(alco))

    for smoke in [0, 1]:
        predicate = lambda df: (df.smoke == smoke)
        add_perturbations(perturbations, predicate, "smoke=="+str(smoke))

    for gender in [1, 2]:
        for alco in [0, 1]:
            predicate = lambda df: (df.gender == gender) & (df.alco == alco)
            add_perturbations(perturbations, predicate, "gender=="+str(gender)+"&aloc=="+str(alco))
            #print(gender, alco, len(test_data[(test_data.gender == gender) & (test_data.alco == alco)]))

    for gender in [1, 2]:
        for smoke in [0, 1]:
            predicate = lambda df: (df.gender == gender) & (df.smoke == smoke)
            add_perturbations(perturbations, predicate, "gender=="+str(gender)+"&smoke=="+str(smoke))
            #print(gender, smoke, len(test_data[(test_data.gender == gender) & (test_data.smoke == smoke)]))

    for alco in [0, 1]:
        for smoke in [0, 1]:
            predicate = lambda df: (df.alco == alco) & (df.smoke == smoke)
            add_perturbations(perturbations, predicate, "smoke=="+str(smoke)+"&aloc=="+str(alco))


    for gender in [1, 2]:
        for smoke in [0, 1]:
            for alco in [0, 1]:
                predicate = lambda df: (df.gender == gender) & (df.alco == alco) & (df.smoke == smoke)
                add_perturbations(perturbations, predicate, "gender=="+str(gender)+"&aloc=="+str(alco)+"&smoke=="+str(smoke))
                #print(gender, smoke, alco,
                #      len(test_data[(test_data.gender == gender) & (test_data.smoke == smoke) &
                #                    (test_data.alco == alco)]))


#    for marital_status in marital_statuses:
#        predicate = lambda df: (df.marital_status == marital_status)
#        add_perturbations(perturbations, predicate)

#    for education in educations:
#        predicate = lambda df: (df.education == education)
#        add_perturbations(perturbations, predicate)

#    for marital_status in marital_statuses:
#        for education in educations:
#            predicate = lambda df: (df.marital_status == marital_status) & (df.education == education)
#            add_perturbations(perturbations, predicate)


            #print(len(test_data))
            #test_slice = test_data[predicate(test_data)]
            #target_slice = target_data[predicate(target_data)]

            #print(marital_status, education, len(test_slice), len(target_slice))

    performance_predictor = train_random_forest_regressor(test_data, y_test, perturbations, model, scoring)

    evaluate_regressor_sliced(target_data, y_target, perturbations, model, performance_predictor, scoring,
                              scoring_name, dataset_name, 'slices', learner_name, experiment_name)
