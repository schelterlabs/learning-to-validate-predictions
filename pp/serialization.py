import warnings
# needed to get rid of deprecation warnings
import imp
from joblib import load
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error
from sklearn.exceptions import DataConversionWarning
import tempfile
import keras.models
import numpy as np

warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=DataConversionWarning)


def make_keras_picklable():
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = {'model_str': model_str}
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


def as_array(_str):
    return np.array([float(elem) for elem in _str.replace("[", "").replace("]", "").split(", ")])


def load_black_box(_model_directory_name):
    _model_directory = './black-box-models/' + _model_directory_name + '/'
    make_keras_picklable()
    _model = load(_model_directory + 'model.joblib')
    _train_data = pd.read_csv(_model_directory + 'train.csv', sep='\t')
    _y_train = load(_model_directory + 'y_train.joblib')
    _test_data = pd.read_csv(_model_directory + 'test.csv', sep='\t')
    _y_test = load(_model_directory + 'y_test.joblib')
    _target_data = pd.read_csv(_model_directory + 'target.csv', sep='\t')
    _y_target = load(_model_directory + 'y_target.joblib')

    _learner_name, _dataset_name, _ = _model_directory_name.split("-")

    if 'convnet' in _model_directory or 'autokeras' in _model_directory:
        _train_data.image = _train_data.apply(lambda row: as_array(row.image), axis=1)
        _test_data.image = _test_data.apply(lambda row: as_array(row.image), axis=1)
        _target_data.image = _target_data.apply(lambda row: as_array(row.image), axis=1)

    if 'accuracy' in _model_directory:
        _scoring_name = 'accuracy'
        _scoring = accuracy_score
    elif 'mae' in _model_directory:
        _scoring_name = 'mae'
        _scoring = mean_absolute_error
    else:
        _scoring_name = 'roc_auc'
        _scoring = roc_auc_score

    return _model, _scoring, _scoring_name, _train_data, _y_train, _test_data, _y_test, _target_data, _y_target, \
        _learner_name, _dataset_name

