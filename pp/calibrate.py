from sklearn.metrics import brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
from pp.serialization import load_black_box


class CalibratedModel:

    def __init__(self, _transformer, calibrated_clf):
        self.transformer = _transformer
        self.calibrated_clf = calibrated_clf

        super().__init__()

    def predict(self, X):
        X_transformed = self.transformer.transform(X)
        return self.calibrated_clf.predict(X_transformed)

    def predict_proba(self, X):
        X_transformed = self.transformer.transform(X)
        return self.calibrated_clf.predict_proba(X_transformed)

    def estimate_accuracy(self, X):
        probas = self.predict_proba(X)
        return np.mean(np.amax(probas, axis=1))


class CalibratedDNNModel:

    def __init__(self, _transformer1, _transformer2, calibrated_clf):
        self.transformer1 = _transformer1
        self.transformer2 = _transformer2
        self.calibrated_clf = calibrated_clf

        super().__init__()

    def predict(self, X):
        X_transformed = self.transformer2.transform(self.transformer1.transform(X))
        return self.calibrated_clf.predict(X_transformed)

    def predict_proba(self, X):
        X_transformed = self.transformer2.transform(self.transformer1.transform(X))
        return self.calibrated_clf.predict_proba(X_transformed)

    def estimate_accuracy(self, X):
        probas = self.predict_proba(X)
        return np.mean(np.amax(probas, axis=1))


#def calibrate(learner_name, model, target_data, y_target, test_data, y_test, method):
def calibrate(learner_name, model, test_data, y_test, method):

    print("\n\tBrier scores: (the smaller the better)")

    prob_pos_clf = model.predict_proba(test_data)[:, 1]
    clf_score = brier_score_loss(y_test, prob_pos_clf)
    print("\tNo calibration: %1.5f" % clf_score)

    if learner_name == 'dnn':

        _, transformer1 = model.best_estimator_.steps[0]
        _, transformer2 = model.best_estimator_.steps[1]
        _, clf = model.best_estimator_.steps[2]

        X_test = transformer2.transform(transformer1.transform(test_data))
        X_target = transformer2.transform(transformer1.transform(test_data))

        clf_sigmoid = CalibratedClassifierCV(clf, cv='prefit', method=method)
        clf_sigmoid.fit(X_test, y_test)

        prob_pos_sigmoid = clf_sigmoid.predict_proba(X_target)[:, 1]
        clf_sigmoid_score = brier_score_loss(y_test, prob_pos_sigmoid)
        print("\tWith %s calibration: %1.5f" % (method, clf_sigmoid_score))
        print("")

        calibrated_model = CalibratedDNNModel(transformer1, transformer2, clf_sigmoid)
        # print(calibrated_model.estimate_accuracy(target_data))
        return calibrated_model


    else:
        _, transformer = model.best_estimator_.steps[0]
        _, clf = model.best_estimator_.steps[1]

        X_test = transformer.transform(test_data)
        X_target = transformer.transform(test_data)

        clf_sigmoid = CalibratedClassifierCV(clf, cv='prefit', method=method)
        clf_sigmoid.fit(X_test, y_test)

        prob_pos_sigmoid = clf_sigmoid.predict_proba(X_target)[:, 1]
        clf_sigmoid_score = brier_score_loss(y_test, prob_pos_sigmoid)
        print("\tWith %s calibration: %1.5f" % (method, clf_sigmoid_score))
        print("")

        calibrated_model = CalibratedModel(transformer, clf_sigmoid)
        # print(calibrated_model.estimate_accuracy(target_data))
        return calibrated_model
