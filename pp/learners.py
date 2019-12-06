from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
from mlxtend.preprocessing import DenseTransformer
from keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Reshape
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import warnings
from sklearn.linear_model import Ridge
warnings.filterwarnings('ignore')
import autosklearn.classification

class Learner(object):

    def __init__(self, scoring):
        self.scoring = scoring

    def split(self, data):
        train_data, heldout_data = train_test_split(data, test_size=0.2)
        test_data, target_data = train_test_split(heldout_data, test_size=0.5)

        return train_data, test_data, target_data

    def scoring_name(self):
        return self.scoring

    def score(self, y_true, y_pred):
        if self.scoring == 'accuracy':
            return accuracy_score(y_true, y_pred)

        if self.scoring == 'roc_auc':
            return roc_auc_score(y_true, y_pred)

        if self.scoring == 'neg_mean_squared_error':
            return mean_squared_error(y_true, y_pred)

        if self.scoring == 'neg_mean_absolute_error':
            return mean_absolute_error(y_true, y_pred)

        raise Exception('unknown scoring {}'.format(self.scoring))


class AutoSklearn(Learner):

    def __init__(self, scoring, time_limit):
        super(AutoSklearn, self).__init__(scoring)
        self.name = "autosklearn"
        self.time_limit = time_limit

    def fit(self, dataset, train_data):

        y_train = dataset.labels_from(train_data)

        if len(dataset.textual_columns) > 1:
            raise Exception('Can only handle one textual column at the moment.')

        sparse_threshold = 0.3
        textual_column = []
        if len(dataset.textual_columns) > 0:
            sparse_threshold = 1.0
            textual_column = dataset.textual_columns[0]

        feature_transformation = ColumnTransformer(transformers=[
            ('categorical_features', OneHotEncoder(handle_unknown='ignore'), dataset.categorical_columns),
            ('scaled_numeric', StandardScaler(), dataset.numerical_columns),
            ('textual_features', HashingVectorizer(ngram_range=(1, 3), n_features=100000), textual_column),
        ], sparse_threshold=sparse_threshold)

        learner = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=self.time_limit)

        pipeline = Pipeline([
            ('features', feature_transformation),
            ('learner', learner)])

        model = pipeline.fit(train_data, y_train)

        return model


class AutoSklearn2(Learner):

    def __init__(self, scoring, time_limit):
        super(AutoSklearn2, self).__init__(scoring)
        self.name = "autosklearn2"
        self.time_limit = time_limit

    def fit(self, dataset, train_data):

        y_train = dataset.labels_from(train_data)

        if len(dataset.textual_columns) > 1:
            raise Exception('Can only handle one textual column at the moment.')

        sparse_threshold = 0.3
        textual_column = []
        if len(dataset.textual_columns) > 0:
            sparse_threshold = 1.0
            textual_column = dataset.textual_columns[0]

        categorical_transformation = Pipeline([
            ('impute', SimpleImputer(strategy='constant', fill_value='__unknown')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        feature_transformation = ColumnTransformer(transformers=[
            ('categorical_features', categorical_transformation, dataset.categorical_columns),
            ('scaled_numeric', StandardScaler(), dataset.numerical_columns),
            ('textual_features', HashingVectorizer(ngram_range=(1, 3), n_features=100000), textual_column),
        ], sparse_threshold=sparse_threshold)

        learner = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=self.time_limit)

        pipeline = Pipeline([
            ('features', feature_transformation),
            ('learner', learner)])

        model = pipeline.fit(train_data, y_train)

        return model


class RidgeRegression(Learner):
    def __init__(self, scoring):
        super(RidgeRegression, self).__init__(scoring)
        self.name = "ridge_regression"

    def fit(self, dataset, train_data):

        y_train = dataset.labels_from(train_data)

        if len(dataset.textual_columns) > 1:
            raise Exception('Can only handle one textual column at the moment.')

        sparse_threshold = 0.3
        textual_column = []
        if len(dataset.textual_columns) > 0:
            sparse_threshold = 1.0
            textual_column = dataset.textual_columns[0]

        feature_transformation = ColumnTransformer(transformers=[
            ('categorical_features', OneHotEncoder(handle_unknown='ignore'), dataset.categorical_columns),
            ('scaled_numeric', StandardScaler(), dataset.numerical_columns),
            ('textual_features', HashingVectorizer(ngram_range=(1, 3), n_features=100000), textual_column),
        ], sparse_threshold=sparse_threshold)

        param_grid = {
            # try smaller alphas
            'learner__alpha': [0.0001, 0.001, 0.01, 0.1]
        }

        pipeline = Pipeline([
            ('features', feature_transformation),
            ('learner', Ridge())])

        search = GridSearchCV(pipeline, param_grid, scoring=self.scoring, cv=5, verbose=1, n_jobs=-1)
        model = search.fit(train_data, y_train)

        return model


class LogisticRegression(Learner):

    def __init__(self, scoring):
        super(LogisticRegression, self).__init__(scoring)
        self.name = "logistic_regression"

    def fit(self, dataset, train_data):

        y_train = dataset.labels_from(train_data)

        if len(dataset.textual_columns) > 1:
            raise Exception('Can only handle one textual column at the moment.')

        sparse_threshold = 0.3
        textual_column = []
        if len(dataset.textual_columns) > 0:
            sparse_threshold = 1.0
            textual_column = dataset.textual_columns[0]

        feature_transformation = ColumnTransformer(transformers=[
            ('categorical_features', OneHotEncoder(handle_unknown='ignore'), dataset.categorical_columns),
            ('scaled_numeric', StandardScaler(), dataset.numerical_columns),
            ('textual_features', HashingVectorizer(ngram_range=(1, 3), n_features=100000), textual_column),
        ], sparse_threshold=sparse_threshold)

        param_grid = {
            'learner__loss': ['log'],
            'learner__penalty': ['l2', 'l1', 'elasticnet'],
            'learner__alpha': [0.0001, 0.001, 0.01, 0.1]
        }

        pipeline = Pipeline([
            ('features', feature_transformation),
            ('learner', SGDClassifier(max_iter=1000))])

        search = GridSearchCV(pipeline, param_grid, scoring=self.scoring, cv=5, verbose=1, n_jobs=-1)
        model = search.fit(train_data, y_train)

        return model


class LassoLogisticRegression(Learner):

    def __init__(self, scoring):
        super(LassoLogisticRegression, self).__init__(scoring)
        self.name = "logistic_regression"

    def fit(self, dataset, train_data):

        y_train = dataset.labels_from(train_data)

        if len(dataset.textual_columns) > 1:
            raise Exception('Can only handle one textual column at the moment.')

        sparse_threshold = 0.3
        textual_column = []
        if len(dataset.textual_columns) > 0:
            sparse_threshold = 1.0
            textual_column = dataset.textual_columns[0]

        feature_transformation = ColumnTransformer(transformers=[
            ('categorical_features', OneHotEncoder(handle_unknown='ignore'), dataset.categorical_columns),
            ('scaled_numeric', StandardScaler(), dataset.numerical_columns),
            ('textual_features', HashingVectorizer(ngram_range=(1, 3), n_features=100000), textual_column),
        ], sparse_threshold=sparse_threshold)

        param_grid = {
            'learner__loss': ['log'],
            'learner__penalty': ['l1'],
            'learner__alpha': [0.0001, 0.001, 0.01, 0.1]
        }

        pipeline = Pipeline([
            ('features', feature_transformation),
            ('learner', SGDClassifier(max_iter=1000))])

        search = GridSearchCV(pipeline, param_grid, scoring=self.scoring, cv=5, verbose=1, n_jobs=-1)
        model = search.fit(train_data, y_train)

        return model




class XgBoost(Learner):

    def __init__(self, scoring):
        super(XgBoost, self).__init__(scoring)
        self.name = "xgboost"

    def fit(self, dataset, train_data):

        y_train = dataset.labels_from(train_data)

        if len(dataset.textual_columns) > 1:
            raise Exception('Can only handle one textual column at the moment.')

        sparse_threshold = 0.3
        textual_column = []
        if len(dataset.textual_columns) > 0:
            sparse_threshold = 1.0
            textual_column = dataset.textual_columns[0]

        feature_transformation = ColumnTransformer(transformers=[
            ('categorical_features', OneHotEncoder(handle_unknown='ignore'), dataset.categorical_columns),
            ('scaled_numeric', StandardScaler(), dataset.numerical_columns),
            ('textual_features', HashingVectorizer(ngram_range=(1, 3), n_features=100000), textual_column),
        ], sparse_threshold=sparse_threshold)

        if self.scoring == 'accuracy':
            xg_metric = 'error'

        if self.scoring == 'roc_auc':
            xg_metric = 'auc'

        param_grid = {
            'learner__n_estimators': [5, 10],
            'learner__max_depth': [3, 6, 10],
            'learner__objective': ['binary:logistic'],
            'learner__eval_metric': [xg_metric]
        }

        pipeline = Pipeline([
            ('features', feature_transformation),
            ('learner', xgb.XGBClassifier())])

        search = GridSearchCV(pipeline, param_grid, scoring=self.scoring, cv=5, verbose=1, n_jobs=-1)
        model = search.fit(train_data, y_train)

        return model


# class XgBoostRegressor(Learner):
#
#     def __init__(self, scoring):
#         super(XgBoostRegressor, self).__init__(scoring)
#         self.name = "xgboostreg"
#
#     def fit(self, dataset, train_data):
#
#         y_train = dataset.labels_from(train_data)
#
#         if len(dataset.textual_columns) > 1:
#             raise Exception('Can only handle one textual column at the moment.')
#
#         sparse_threshold = 0.3
#         textual_column = []
#         if len(dataset.textual_columns) > 0:
#             sparse_threshold = 1.0
#             textual_column = dataset.textual_columns[0]
#
#         feature_transformation = ColumnTransformer(transformers=[
#             ('categorical_features', OneHotEncoder(handle_unknown='ignore'), dataset.categorical_columns),
#             ('scaled_numeric', StandardScaler(), dataset.numerical_columns),
#             ('textual_features', HashingVectorizer(ngram_range=(1, 3), n_features=100000), textual_column),
#         ], sparse_threshold=sparse_threshold)
#
#         param_grid = {
#             'learner__n_estimators': [5, 10],
#             'learner__max_depth': [3, 6, 10],
#             #'learner__objective': ['reg:squarederror'],
#         }
#
#         pipeline = Pipeline([
#             ('features', feature_transformation),
#             ('learner', xgb.XGBRegressor())])
#
#         search = GridSearchCV(pipeline, param_grid, scoring=self.scoring, cv=5, verbose=1, n_jobs=-1)
#         model = search.fit(train_data, y_train)
#
#         return model


import types
import tempfile
import keras.models


def make_keras_picklable():
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = { 'model_str': model_str }
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = keras.models.load_model(fd.name)
        self.__dict__ = model.__dict__

    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__


class DNN(Learner):

    def __init__(self, scoring):
        super(DNN, self).__init__(scoring)
        self.name = "dnn"

    @staticmethod
    def create_model(size_1, size_2):
        nn = keras.Sequential([
            keras.layers.Dense(size_1, activation=tf.nn.relu),
            keras.layers.Dense(size_2, activation=tf.nn.relu),
            keras.layers.Dense(2, activation=tf.nn.softmax)
        ])

        nn.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.005),
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy']) # TODO figure out how to use roc_auc here...
        return nn

    def fit(self, dataset, train_data):

        y_train = dataset.labels_from(train_data)

        # feature_transformation = ColumnTransformer(transformers=[
        #     ('categorical_features', OneHotEncoder(handle_unknown='ignore'), dataset.categorical_columns),
        #     ('scaled_numeric', StandardScaler(), dataset.numerical_columns)
        # ], sparse_threshold=0)

        if len(dataset.textual_columns) > 1:
            raise Exception('Can only handle one textual column at the moment.')

        sparse_threshold = 0.3
        textual_column = []
        if len(dataset.textual_columns) > 0:
            sparse_threshold = 0.0
            textual_column = dataset.textual_columns[0]

        feature_transformation = ColumnTransformer(transformers=[
            ('categorical_features', OneHotEncoder(handle_unknown='ignore'), dataset.categorical_columns),
            ('scaled_numeric', StandardScaler(), dataset.numerical_columns),
            ('textual_features', HashingVectorizer(ngram_range=(1, 3), n_features=10000), textual_column),
        ], sparse_threshold=sparse_threshold)

        make_keras_picklable()
        nn_model = keras.wrappers.scikit_learn.KerasClassifier(build_fn=self.create_model)

        pipeline = Pipeline([
            ('features', feature_transformation),
            ('todense', DenseTransformer()),
            ('learner', nn_model)])

        param_grid = {
            'learner__epochs': [50],
            'learner__batch_size': [1024],
            'learner__size_1': [4, 8],
            'learner__size_2': [2, 4],
            'learner__verbose': [1]
        }

        model = GridSearchCV(pipeline, param_grid, scoring=self.scoring, cv=5, verbose=2).fit(train_data, y_train)

        return model


class ArrayTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, input_array, y=None):
        return self

    @staticmethod
    def transform(inputs, y=None):
        column_name = inputs.columns.tolist()[0]
        output = np.array([np.array(elem) for elem in np.array(inputs[column_name])])
        print(output.shape)
        return output


class ArrayTransformer3D(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, input_array, y=None):
        return self

    @staticmethod
    def transform(inputs):
        output = np.array([np.array(elem) for elem in np.array(inputs)])
        num_samples = output.shape[0]
        output = output.reshape((num_samples, 28, 28))
        # print(output.shape)
        return output


class ConvNet(Learner):

    def __init__(self, scoring):
        super(ConvNet, self).__init__(scoring)
        self.name = "convnet"

    @staticmethod
    def create_model():
        input_shape = (28, 28, 1)
        _model = Sequential()
        _model.add(Reshape((28, 28, 1,), input_shape=(784,)))
        _model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        _model.add(Conv2D(64, (3, 3), activation='relu'))
        _model.add(MaxPooling2D(pool_size=(2, 2)))
        _model.add(Dropout(0.25))
        _model.add(Flatten())
        _model.add(Dense(128, activation='relu'))
        _model.add(Dropout(0.5))
        _model.add(Dense(2, activation='softmax'))

        _model.compile(loss=keras.losses.categorical_crossentropy,
                       optimizer=keras.optimizers.Adadelta(),
                       metrics=['accuracy'])

        return _model

    def fit(self, dataset, train_data):

        y_train = dataset.labels_from(train_data)

        make_keras_picklable()
        convnet = keras.wrappers.scikit_learn.KerasClassifier(build_fn=self.create_model, epochs=10, batch_size=128)

        feature_transformation = ColumnTransformer(transformers=[
            ('projection', ArrayTransformer(), ['image'])
        ])

        pipeline = Pipeline([
            ('features', feature_transformation),
            ('learner', convnet)])

        return pipeline.fit(train_data, y_train)


class AutoKerasModel:

    def __init__(self, clf):
        super().__init__()
        self.clf = clf

    def predict(self, data):
        return self.clf.predict(ArrayTransformer3D().transform(data['image']))

    def predict_proba(self, data):
        x_test = ArrayTransformer3D().transform(data['image'])
        x_test = self.clf.preprocess(x_test)
        test_loader = self.clf.data_transformer.transform_test(x_test)
        return self.clf.cnn.predict(test_loader)


from autokeras import ImageClassifier
class AutoKeras(Learner):

    def __init__(self, scoring):
        super(AutoKeras, self).__init__(scoring)
        self.name = "autokeras"

    def fit(self, dataset, train_data):

        y_train = dataset.labels_from(train_data)

        make_keras_picklable()

        train_data_transformed = ArrayTransformer3D().transform(train_data['image'])
        clf = ImageClassifier(verbose=True, augment=False)
        clf.fit(train_data_transformed, y_train, time_limit=120)

        return AutoKerasModel(clf)


class AutoKerasLarge(Learner):

    def __init__(self, scoring):
        super(AutoKerasLarge, self).__init__(scoring)
        self.name = "autokeraslarge"

    def fit(self, dataset, train_data):

        y_train = dataset.labels_from(train_data)

        make_keras_picklable()

        train_data_transformed = ArrayTransformer3D().transform(train_data['image'])

        searcher_args = {'default_model_len': 10}

        clf = ImageClassifier(verbose=True, augment=False, searcher_args=searcher_args)
        clf.fit(train_data_transformed, y_train, time_limit=120)

        return AutoKerasModel(clf)


from tpot import TPOTClassifier
class TPOT(Learner):

    def __init__(self, scoring):
        super(TPOT, self).__init__(scoring)
        self.name = "tpot"

    def fit(self, dataset, train_data):

        y_train = dataset.labels_from(train_data)

        if len(dataset.textual_columns) > 1:
            raise Exception('Can only handle one textual column at the moment.')

        sparse_threshold = 0.3
        textual_column = []
        if len(dataset.textual_columns) > 0:
            sparse_threshold = 1.0
            textual_column = dataset.textual_columns[0]

        feature_transformation = ColumnTransformer(transformers=[
            ('categorical_features', OneHotEncoder(handle_unknown='ignore'), dataset.categorical_columns),
            ('scaled_numeric', StandardScaler(), dataset.numerical_columns),
            ('textual_features', HashingVectorizer(ngram_range=(1, 3), n_features=100000), textual_column),
        ], sparse_threshold=sparse_threshold)

        param_grid = {
            'learner__loss': ['log'],
            'learner__penalty': ['l2', 'l1', 'elasticnet'],
            'learner__alpha': [0.0001, 0.001, 0.01, 0.1]
        }

        optimizer = TPOTClassifier(generations=5, population_size=20, cv=5, random_state=42, verbosity=2,
                                   config_dict='TPOT sparse', max_time_mins=2)

        pipeline = Pipeline([
            ('features', feature_transformation),
            ('learner', optimizer)])

        # search = GridSearchCV(pipeline, param_grid, scoring=self.scoring, cv=5, verbose=1, n_jobs=-1)
        model = pipeline.fit(train_data, y_train)

        return model
