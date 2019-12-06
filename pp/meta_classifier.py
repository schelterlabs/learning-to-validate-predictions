from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
import xgboost as xgb
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from scipy import stats
from pp.baselines import relational_shift


def train_performance_predictor(test_data, y_test, threshold, perturbations_to_apply, model, scoring):

    score_on_test_data = scoring(y_test, model.predict(test_data))
    predictions_on_test_data = model.predict_proba(test_data)
    test_probs_class_a = np.transpose(predictions_on_test_data)[0]

    num_pos_test = np.array(y_test).sum()
    num_neg_test = len(y_test) - num_pos_test

    print("Generating perturbed training data...")
    generated_training_data = []
    generated_labels = []

    for perturbation in perturbations_to_apply:
        corrupted_test_data = perturbation.transform(test_data)

        predictions = model.predict_proba(corrupted_test_data)

        probs_class_a = np.transpose(predictions)[0]
        features = np.percentile(probs_class_a, np.arange(0, 101, 1))
        _, bbse_soft_p_val = stats.ks_2samp(test_probs_class_a, probs_class_a)

        y_predicted = model.predict(corrupted_test_data)
        num_pos_predicted = np.array(y_predicted).sum()
        num_neg_predicted = len(y_predicted) - num_pos_predicted

        _, bbse_hard_p_val, _, _ = stats.chi2_contingency([[num_pos_predicted, num_neg_predicted],
                                                           [num_pos_test, num_neg_test]])

        generated_training_data.append(np.concatenate(([bbse_soft_p_val, bbse_hard_p_val], features)))

        score_on_corrupted_test_data = scoring(y_test, model.predict(corrupted_test_data))
        decline = 1.0 - (score_on_corrupted_test_data / score_on_test_data)

        if decline <= threshold:
            generated_labels.append(1)
        else:
            generated_labels.append(0)

        if len(generated_labels) % 100 == 0:
            print('{}/{} examples generated...'.format(len(generated_labels), len(perturbations_to_apply)))

    X = np.array(generated_training_data)
    y = np.array(generated_labels)

    param_grid = {
        'learner__n_estimators': [5, 10, 20, 50],
        'learner__max_depth': [3, 6, 10],
        'learner__objective': ['binary:logistic'],
        'learner__eval_metric': ['auc']
    }

    meta_regressor_pipeline = Pipeline([
        ('learner', xgb.XGBClassifier())])

    print("Training performance predictor...")
    meta_regressor = GridSearchCV(meta_regressor_pipeline, param_grid, cv=5, scoring='accuracy').fit(X, y)

    print("Done with training")
    return meta_regressor


def evaluate_predictor(target_data, y_target, threshold, perturbations_to_apply, model, meta_regressor, scoring,
                       scoring_name, dataset_name, perturbations_name, learner_name, experiment_name, test_data,
                       y_test, categorical_columns, numerical_columns):

    num_pos_test = np.array(y_test).sum()
    num_neg_test = len(y_test) - num_pos_test

    score_on_target_data = scoring(y_target, model.predict(target_data))
    predictions_on_test_data = model.predict_proba(test_data)
    test_probs_class_a = np.transpose(predictions_on_test_data)[0]

    true_labels = []
    predicted_labels = []
    bbse_soft_predicted_labels = []
    bbse_hard_predicted_labels = []
    relational_shift_predicted_labels = []

    print("Evaluating on perturbed test data...")
    for perturbation in perturbations_to_apply:
        corrupted_target_data = perturbation.transform(target_data)

        predictions = model.predict_proba(corrupted_target_data)
        probs_class_a = np.transpose(predictions)[0]
        features = np.percentile(probs_class_a, np.arange(0, 101, 1))

        _, bbse_soft_p_val = stats.ks_2samp(test_probs_class_a, probs_class_a)

        y_predicted = model.predict(corrupted_target_data)
        num_pos_predicted = np.array(y_predicted).sum()
        num_neg_predicted = len(y_predicted) - num_pos_predicted

        _, bbse_hard_p_val, _, _ = stats.chi2_contingency([[num_pos_predicted, num_neg_predicted],
                                                           [num_pos_test, num_neg_test]])

        if len(categorical_columns) > 0 or len(numerical_columns) > 0:
            is_raw_shift = relational_shift(test_data, corrupted_target_data, numerical_columns, categorical_columns)
        else:
            # We cannot apply this to images...
            is_raw_shift = False


        score_on_corrupted_target_data = scoring(y_target, model.predict(corrupted_target_data))

        decline = 1.0 - (score_on_corrupted_target_data / score_on_target_data)

        if decline <= threshold:
            true_labels.append(1)
        else:
            true_labels.append(0)

        if bbse_soft_p_val < 0.05:
            bbse_soft_predicted_labels.append(0)
        else:
            bbse_soft_predicted_labels.append(1)

        if bbse_hard_p_val < 0.05:
            bbse_hard_predicted_labels.append(0)
        else:
            bbse_hard_predicted_labels.append(1)

        if is_raw_shift:
            relational_shift_predicted_labels.append(0)
        else:
            relational_shift_predicted_labels.append(1)

        features_for_prediction = np.concatenate(([bbse_soft_p_val, bbse_hard_p_val], features))
        # features_for_prediction = np.concatenate(([bbse_soft_p_val], [bbse_hard_p_val]))
        predicted_labels.append(meta_regressor.predict([features_for_prediction]))

        if len(predicted_labels) % 100 == 0:
            print('{}/{} examples evaluated...'.format(len(predicted_labels), len(perturbations_to_apply)))

    accuracy_ppm = accuracy_score(true_labels, predicted_labels)
    accuracy_bbse = accuracy_score(true_labels, bbse_soft_predicted_labels)
    accuracy_bbseh = accuracy_score(true_labels, bbse_hard_predicted_labels)
    accuracy_rel = accuracy_score(true_labels, relational_shift_predicted_labels)

    auc_ppm = None
    auc_bbse = None
    auc_bbseh = None
    auc_rel = None

    try:
        auc_ppm = roc_auc_score(true_labels, predicted_labels)
        auc_bbse = roc_auc_score(true_labels, bbse_soft_predicted_labels)
        auc_bbseh = roc_auc_score(true_labels, bbse_hard_predicted_labels)
        auc_rel = roc_auc_score(true_labels, relational_shift_predicted_labels)
    except Exception:
        pass

    ppm_tn, ppm_fp, ppm_fn, ppm_tp = confusion_matrix(true_labels, predicted_labels, labels=[0, 1]).ravel()
    bbse_tn, bbse_fp, bbse_fn, bbse_tp = confusion_matrix(true_labels, bbse_soft_predicted_labels, labels=[0, 1]).ravel()
    bbseh_tn, bbseh_fp, bbseh_fn, bbseh_tp = confusion_matrix(true_labels, bbse_hard_predicted_labels, labels=[0, 1]).ravel()
    rel_tn, rel_fp, rel_fn, rel_tp = confusion_matrix(true_labels, relational_shift_predicted_labels, labels=[0, 1]).ravel()

    line = '{},{},{},{},{},{},{},{},{},{},{},{},{}'

    ppm_log = line.format(experiment_name, dataset_name, learner_name, perturbations_name, scoring_name, threshold,
                          'PPM', accuracy_ppm, auc_ppm, ppm_tn, ppm_fp, ppm_fn, ppm_tp)
    bbse_log = line.format(experiment_name, dataset_name, learner_name, perturbations_name, scoring_name, threshold,
                           'BBSE', accuracy_bbse, auc_bbse, bbse_tn, bbse_fp, bbse_fn, bbse_tp)
    bbseh_log = line.format(experiment_name, dataset_name, learner_name, perturbations_name, scoring_name, threshold,
                            'BBSEh', accuracy_bbseh, auc_bbseh, bbseh_tn, bbseh_fp, bbseh_fn, bbseh_tp)
    rel_log = line.format(experiment_name, dataset_name, learner_name, perturbations_name, scoring_name, threshold,
                          'REL', accuracy_rel, auc_rel, rel_tn, rel_fp, rel_fn, rel_tp)

    print('experiment,dataset,learner,perturbation,scoring,threshold,approach,accuracy,roc_auc,true_negatives,' +
          'false_positives,false_negatives,true_positives')
    print(ppm_log)
    print(bbse_log)
    print(bbseh_log)
    print(rel_log)

    import uuid

    path = '/home/ssc/Entwicklung/projects/projects-black-box-performance-prediction/artifacts/classifier-results/{}.txt'.format(uuid.uuid4())

    with open(path, 'w') as file:
        file.write(ppm_log)
        file.write('\n')
        file.write(bbse_log)
        file.write('\n')
        file.write(bbseh_log)
        file.write('\n')
        file.write(rel_log)
        file.write('\n')
