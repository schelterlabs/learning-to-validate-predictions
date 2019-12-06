import warnings
# needed to get rid of deprecation warnings
import imp
from joblib import dump
from pp import datasets
from pp import learners
from sklearn.exceptions import DataConversionWarning
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import operator
import random
import string

warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

dataset = datasets.MinimalBalancedAdultDataset()
learner = learners.LassoLogisticRegression('accuracy')

model_directory = 'black-box-models/lasso-adult_minimal-accuracy/'

train_data, test_data, target_data = learner.split(dataset.df)

encoder = OneHotEncoder(handle_unknown='ignore').fit(np.array(train_data['workclass']).reshape(-1, 1))

values = train_data['workclass']
encoded = encoder.transform(np.array(train_data['workclass']).reshape(-1, 1))

encoded_a = encoder.transform(np.array(test_data['workclass']).reshape(-1, 1)).todense()
encoded_b = encoder.transform(np.array(target_data['workclass']).reshape(-1, 1)).todense()

categories = {}
for indexed_category, coefficient in zip(values, encoded):
    vector = np.asarray(coefficient[0, :].todense()).reshape(-1)
    index = np.where(vector == 1)[0][0]
    categories[indexed_category] = index

sorted_categories = sorted(categories.items(), key=operator.itemgetter(1))

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

coefficients = model.best_estimator_.named_steps['learner'].coef_

print()
assigned_coefficients = coefficients[:len(sorted_categories)].reshape(-1)

ignored_values = []

for (indexed_category, coefficient) in zip(sorted_categories, assigned_coefficients):
    if coefficient == 0:
        print('workclass', indexed_category[0], 'is ignored by classifier')
        ignored_values.append(indexed_category[0])

test_data_copy = test_data.copy(deep=True)
target_data_copy = target_data.copy(deep=True)

print(len(test_data_copy[test_data_copy['workclass'].isin(ignored_values)]), len(test_data_copy), 'in test data')
print(len(target_data_copy[target_data_copy['workclass'].isin(ignored_values)]), len(target_data_copy), 'target data')


def random_string():
    return ''.join([random.choice(string.ascii_letters + string.digits) for _ in range(0, 32)])


test_data_copy.loc[test_data_copy['workclass'].isin(ignored_values), 'workclass'] = random_string()
target_data_copy.loc[target_data_copy['workclass'].isin(ignored_values), 'workclass'] = random_string()

score_on_corrupted_test_data = learner.score(y_test, model.predict(test_data_copy))
score_on_corrupted_target_data = learner.score(y_target, model.predict(target_data_copy))

print()

print(learner.scoring, "on corrupted test data: ", score_on_corrupted_test_data)
print(learner.scoring, "on corrupted target data: ", score_on_corrupted_target_data)


dump(model, model_directory + 'model.joblib')
dump(y_train, model_directory + 'y_train.joblib')
dump(y_test, model_directory + 'y_test.joblib')
dump(y_target, model_directory + 'y_target.joblib')

train_data.to_csv(model_directory + 'train.csv', sep='\t', encoding='utf-8')
test_data.to_csv(model_directory + 'test.csv', sep='\t', encoding='utf-8')
target_data.to_csv(model_directory + 'target.csv', sep='\t', encoding='utf-8')
