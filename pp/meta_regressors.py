from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt
import numpy as np
import os
from pp.calibrate import calibrate
import pandas as pd
import datetime


def compute_features(predictions):
    probs_class_a = np.transpose(predictions)[0]
    probs_class_b = np.transpose(predictions)[1]
    features_a = np.percentile(probs_class_a, np.arange(0, 101, 5))
    features_b = np.percentile(probs_class_b, np.arange(0, 101, 5))
    return np.concatenate((features_a, features_b), axis=0)


def compute_regression_features(predictions):
    return np.percentile(predictions, np.arange(0, 101, 5))


def train_random_forest_regressor(test_data, y_test, perturbations_to_apply, model, scoring):

    print("Generating perturbed training data...")
    num_examples = 0
    generated_training_data = []

    for perturbation in perturbations_to_apply:
        corrupted_test_data = perturbation.transform(test_data)
        try:
            predictions = model.predict_proba(corrupted_test_data)
            features = compute_features(predictions)
            num_features = 42
        except AttributeError:
            print("No predict_proba...")
            predictions = model.predict(corrupted_test_data)
            features = compute_regression_features(predictions)
            num_features = 21

        score_on_corrupted_test_data = scoring(y_test, model.predict(corrupted_test_data))

        example = np.concatenate((features, [score_on_corrupted_test_data]), axis=0)

        generated_training_data.append(example)
        num_examples += 1
        if num_examples % 10 == 0:
            print(num_examples, len(perturbations_to_apply))

#    f = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../artifacts/intermediates/train.txt')

#    np.savetxt(f, np.array(generated_training_data), delimiter=',')

    X = np.array(generated_training_data)[:, :num_features]
    y = np.array(generated_training_data)[:, num_features]

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
    return meta_regressor


def train_random_forest_regressor_with_noise(test_data, y_test, how_often, model,  scoring):

    print("Generating perturbed training data...")
    num_examples = 0
    generated_training_data = []


    _, transformer = model.best_estimator_.steps[0]
    _, clf = model.best_estimator_.steps[1]

    for _ in range(0, how_often):

        featurized_test_data = transformer.transform(test_data)
        noise = np.random.normal(0, np.random.randint(10), featurized_test_data.shape)

        corrupted_test_data = featurized_test_data + noise

        try:
            #predictions = model.predict_proba(corrupted_test_data)
            predictions = clf.predict_proba(corrupted_test_data)
            features = compute_features(predictions)
            num_features = 42
        except AttributeError:
            print("No predict_proba...")
            #predictions = model.predict(corrupted_test_data)
            predictions = clf.predict(corrupted_test_data)
            features = compute_regression_features(predictions)
            num_features = 21

        #score_on_corrupted_test_data = scoring(y_test, model.predict(corrupted_test_data))
        score_on_corrupted_test_data = scoring(y_test, clf.predict(corrupted_test_data))

        example = np.concatenate((features, [score_on_corrupted_test_data]), axis=0)

        generated_training_data.append(example)
        num_examples += 1
        if num_examples % 10 == 0:
            print(num_examples, '/', how_often)

#    f = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../artifacts/intermediates/train.txt')

#    np.savetxt(f, np.array(generated_training_data), delimiter=',')

    X = np.array(generated_training_data)[:, :num_features]
    y = np.array(generated_training_data)[:, num_features]

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
    return meta_regressor


def evaluate_regressor_with_noise(target_data, y_target, how_often, model, meta_regressor, scoring, scoring_name,
                       dataset_name, perturbations_name, learner_name, experiment_name):

    num_examples = 0

    predicted_scores = []
    true_scores = []

    _, transformer = model.best_estimator_.steps[0]
    _, clf = model.best_estimator_.steps[1]


    print("Evaluating on perturbed test data...")
    for _ in range(0, how_often):

        featurized_target_data = transformer.transform(target_data)
        noise = np.random.normal(0, np.random.randint(10), featurized_target_data.shape)

        corrupted_target_data = featurized_target_data + noise

        try:
            predictions = clf.predict_proba(corrupted_target_data)
            features = compute_features(predictions)
        except AttributeError:
            print("No predict_proba...")
            predictions = clf.predict(corrupted_target_data)
            features = compute_regression_features(predictions)

        score_on_corrupted_target_data = scoring(y_target, clf.predict(corrupted_target_data))
        predicted_score_on_corrupted_target_data = meta_regressor.predict([features])

        predicted_scores.append(predicted_score_on_corrupted_target_data)
        true_scores.append(score_on_corrupted_target_data)

        num_examples += 1
        if num_examples % 10 == 0:
            print(num_examples, '/', how_often)

    plt.plot([0, 1], [0, 1], '-', color='grey', alpha=0.5)

    min_score = np.min(predicted_scores + true_scores) - 0.05
    max_score = np.max(predicted_scores + true_scores) + 0.05

    plt.scatter(true_scores, predicted_scores, alpha=0.05)

    plt.xlabel("true " + scoring_name, fontsize=18)
    plt.ylabel("predicted " + scoring_name, fontsize=18)

    plt.xlim((min_score, max_score))
    plt.ylim((min_score, max_score))

    from sklearn.metrics import mean_squared_error, mean_absolute_error

    mse = mean_squared_error(true_scores, predicted_scores)
    mae = mean_absolute_error(true_scores, predicted_scores)

    text_x = min_score + ((max_score - min_score) / 3.0)
    text_y = min_score + ((max_score - min_score) / 10.0)

    plt.text(text_x, text_y, "MSE %.5f   MAE %.4f" % (mse, mae), fontsize=12,
             bbox=dict(facecolor='none', edgecolor='black', pad=10.0))

    print("MSE %.5f, MAE %.4f" % (mse, mae))

    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tick_params(axis='both', which='minor', labelsize=16)
    plt.legend(['perfect', 'predicted'], fontsize=18)
    plt.gcf().set_size_inches(6, 5)

    experiment_identifier = "__".join([experiment_name, dataset_name, perturbations_name, learner_name, scoring_name])

    plot_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             '../artifacts/figures/' + experiment_identifier + '.pdf')

    print("Writing plot to " + plot_file)
    plt.tight_layout()
    plt.gcf().savefig(plot_file, dpi=300)

    plt.clf()
    plt.cla()
    plt.close()

    results_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                '../artifacts/results/' + experiment_identifier + ".tsv")
    print(results_file)

    with open(results_file, 'w') as the_file:
        the_file.write('y_true\ty_pred\n')
        for true_score, predicted_score in zip(true_scores, predicted_scores):
            the_file.write('%s\t%s\n' % (true_score, predicted_score[0]))

    return mse, mae, plot_file




def evaluate_regressor(target_data, y_target, perturbations_to_apply, model, meta_regressor, scoring, scoring_name,
                       dataset_name, perturbations_name, learner_name, experiment_name):

    num_examples = 0

    predicted_scores = []
    true_scores = []

    print("Evaluating on perturbed test data...")
    for perturbation in perturbations_to_apply:
        corrupted_target_data = perturbation.transform(target_data)

        try:
            predictions = model.predict_proba(corrupted_target_data)
            features = compute_features(predictions)
        except AttributeError:
            print("No predict_proba...")
            predictions = model.predict(corrupted_target_data)
            features = compute_regression_features(predictions)

        score_on_corrupted_target_data = scoring(y_target, model.predict(corrupted_target_data))
        predicted_score_on_corrupted_target_data = meta_regressor.predict([features])

        predicted_scores.append(predicted_score_on_corrupted_target_data)
        true_scores.append(score_on_corrupted_target_data)

        num_examples += 1
        if num_examples % 10 == 0:
            print(num_examples, '/', len(perturbations_to_apply))

    plt.plot([0, 1], [0, 1], '-', color='grey', alpha=0.5)

    min_score = np.min(predicted_scores + true_scores) - 0.05
    max_score = np.max(predicted_scores + true_scores) + 0.05

    plt.scatter(true_scores, predicted_scores, alpha=0.05)

    plt.xlabel("true " + scoring_name, fontsize=18)
    plt.ylabel("predicted " + scoring_name, fontsize=18)

    plt.xlim((min_score, max_score))
    plt.ylim((min_score, max_score))

    from sklearn.metrics import mean_squared_error, mean_absolute_error

    mse = mean_squared_error(true_scores, predicted_scores)
    mae = mean_absolute_error(true_scores, predicted_scores)

    text_x = min_score + ((max_score - min_score) / 3.0)
    text_y = min_score + ((max_score - min_score) / 10.0)

    plt.text(text_x, text_y, "MSE %.5f   MAE %.4f" % (mse, mae), fontsize=12,
             bbox=dict(facecolor='none', edgecolor='black', pad=10.0))

    print("MSE %.5f, MAE %.4f" % (mse, mae))

    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tick_params(axis='both', which='minor', labelsize=16)
    plt.legend(['perfect', 'predicted'], fontsize=18)
    plt.gcf().set_size_inches(6, 5)

    experiment_identifier = "__".join([experiment_name, dataset_name, perturbations_name, learner_name, scoring_name])

    plot_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             '../artifacts/figures/' + experiment_identifier + '.pdf')

    print("Writing plot to " + plot_file)
    plt.tight_layout()
    plt.gcf().savefig(plot_file, dpi=300)

    plt.clf()
    plt.cla()
    plt.close()

    results_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                '../artifacts/results/' + experiment_identifier + ".tsv")
    print(results_file)

    with open(results_file, 'w') as the_file:
        the_file.write('y_true\ty_pred\n')
        for true_score, predicted_score in zip(true_scores, predicted_scores):
            the_file.write('%s\t%s\n' % (true_score, predicted_score[0]))

    return mse, mae, plot_file


def evaluate_regressor_sliced(target_data, y_target, perturbations_to_apply, model, meta_regressor, scoring, scoring_name,
                              dataset_name, perturbations_name, learner_name, experiment_name):

    predicted_scores = []
    true_scores = []
    infos = []

    print("Evaluating on perturbed test data...")
    for perturbation in perturbations_to_apply:
        corrupted_target_data = perturbation.transform(target_data)

        try:
            predictions = model.predict_proba(corrupted_target_data)
            features = compute_features(predictions)
        except AttributeError:
            print("No predict_proba...")
            predictions = model.predict(corrupted_target_data)
            features = compute_regression_features(predictions)

        score_on_corrupted_target_data = scoring(y_target, model.predict(corrupted_target_data))
        predicted_score_on_corrupted_target_data = meta_regressor.predict([features])

        predicted_scores.append(predicted_score_on_corrupted_target_data)
        true_scores.append(score_on_corrupted_target_data)

        infos.append(str(perturbation.num_affected(target_data)) + "\t" + perturbation.name())

    plt.plot([0, 1], [0, 1], '-', color='grey', alpha=0.5)

    min_score = np.min(predicted_scores + true_scores) - 0.05
    max_score = np.max(predicted_scores + true_scores) + 0.05

    plt.scatter(true_scores, predicted_scores, alpha=0.05)

    plt.xlabel("true " + scoring_name, fontsize=18)
    plt.ylabel("predicted " + scoring_name, fontsize=18)

    plt.xlim((min_score, max_score))
    plt.ylim((min_score, max_score))

    from sklearn.metrics import mean_squared_error, mean_absolute_error

    mse = mean_squared_error(true_scores, predicted_scores)
    mae = mean_absolute_error(true_scores, predicted_scores)

    text_x = min_score + ((max_score - min_score) / 3.0)
    text_y = min_score + ((max_score - min_score) / 10.0)

    plt.text(text_x, text_y, "MSE %.5f   MAE %.4f" % (mse, mae), fontsize=12,
             bbox=dict(facecolor='none', edgecolor='black', pad=10.0))

    print("MSE %.5f, MAE %.4f" % (mse, mae))

    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tick_params(axis='both', which='minor', labelsize=16)
    plt.legend(['perfect', 'predicted'], fontsize=18)
    plt.gcf().set_size_inches(6, 5)

    experiment_identifier = "__".join([experiment_name, dataset_name, perturbations_name, learner_name, scoring_name])

    plot_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             '../artifacts/figures/' + experiment_identifier + '.pdf')

    print("Writing plot to " + plot_file)
    plt.tight_layout()
    plt.gcf().savefig(plot_file, dpi=300)

    plt.clf()
    plt.cla()
    plt.close()

    results_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                '../artifacts/results/' + experiment_identifier + ".tsv")
    print(results_file)

    with open(results_file, 'w') as the_file:
        the_file.write('y_true\ty_pred\tslice\tinfo\n')
        for true_score, predicted_score, info in zip(true_scores, predicted_scores, infos):
            the_file.write('%s\t%s\t%s\n' % (true_score, predicted_score[0], info))

    return mse, mae, plot_file



def evaluate_regressor_against_calibration_baseline(test_data, y_test, target_data, y_target, perturbations_to_apply,
                                                    model, meta_regressor, scoring, scoring_name, dataset_name,
                                                    perturbations_name, learner_name, experiment_name,
                                                    calibration_method):

    calibrated_model = calibrate(learner_name, model, target_data, y_target, test_data, y_test, calibration_method)

    predicted_scores = []
    predicted_scores_calibration = []
    true_scores = []

    print("Evaluating on perturbed test data...")
    for perturbation in perturbations_to_apply:
        corrupted_target_data = perturbation.transform(target_data)

        try:
            predictions = model.predict_proba(corrupted_target_data)
            features = compute_features(predictions)
        except AttributeError:
            print("No predict_proba...")
            predictions = model.predict(corrupted_target_data)
            features = compute_regression_features(predictions)

        score_on_corrupted_target_data = scoring(y_target, model.predict(corrupted_target_data))
        predicted_score_on_corrupted_target_data = meta_regressor.predict([features])

        predicted_scores.append(predicted_score_on_corrupted_target_data)
        predicted_scores_calibration.append(calibrated_model.estimate_accuracy(corrupted_target_data))
        true_scores.append(score_on_corrupted_target_data)


    experiment_identifier = "__".join([experiment_name, dataset_name, perturbations_name, learner_name, scoring_name])

    results_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                '../artifacts/calibration/' + experiment_identifier + ".tsv")
    print(results_file)

    with open(results_file, 'w') as the_file:
        the_file.write('y_true\ty_pred\ty_pred_calib\n')
        for true_score, predicted_score, predicted_score_calibration in zip(true_scores, predicted_scores, predicted_scores_calibration):
            the_file.write('%s\t%s\t%s\n' % (true_score, predicted_score[0], predicted_score_calibration))




def evaluate_regressor_against_corrupted_calibration_baseline(test_data, y_test, target_data, y_target,
                                                              perturbations_to_apply, model, meta_regressor, scoring,
                                                              scoring_name, dataset_name, perturbations_name,
                                                              learner_name, experiment_name, calibration_method,
                                                              learner, dataset):
    print("Training calibrated classifier...")

    unioned_corrupted_data = None

    for perturbation in perturbations_to_apply:
        corrupted_test_data = perturbation.transform(test_data)

        if unioned_corrupted_data is None:
            unioned_corrupted_data = corrupted_test_data
        else:
            unioned_corrupted_data = pd.concat([unioned_corrupted_data, corrupted_test_data], ignore_index=True)

    print("NUM SAMPLES", len(unioned_corrupted_data))

    if len(unioned_corrupted_data) > 100000:
        unioned_corrupted_data = unioned_corrupted_data.sample(n=100000)

    model_to_calibrate = learner.fit(dataset, unioned_corrupted_data)
    y_test_calib = dataset.labels_from(unioned_corrupted_data)

    calibrated_model = calibrate(learner_name, model_to_calibrate, unioned_corrupted_data, y_test_calib,
                                 calibration_method)

    predicted_scores = []
    predicted_scores_calibration = []
    true_scores = []

    num_evaluated = 0

    print("Evaluating on perturbed test data...")
    for perturbation in perturbations_to_apply:
        corrupted_target_data = perturbation.transform(target_data)

        try:
            predictions = model.predict_proba(corrupted_target_data)
            features = compute_features(predictions)
        except AttributeError:
            print("No predict_proba...")
            predictions = model.predict(corrupted_target_data)
            features = compute_regression_features(predictions)

        score_on_corrupted_target_data = scoring(y_target, model.predict(corrupted_target_data))
        predicted_score_on_corrupted_target_data = meta_regressor.predict([features])

        predicted_scores.append(predicted_score_on_corrupted_target_data)
        predicted_scores_calibration.append(calibrated_model.estimate_accuracy(corrupted_target_data))
        true_scores.append(score_on_corrupted_target_data)

        num_evaluated += 1
        if num_evaluated % 10 == 0:
            print("\t Evaluation done: ", num_evaluated, "\t", datetime.datetime.now())


    experiment_identifier = "__".join([experiment_name, dataset_name, perturbations_name, learner_name, scoring_name])

    results_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                '../artifacts/calibration/' + experiment_identifier + ".tsv")
    print(results_file)

    with open(results_file, 'w') as the_file:
        the_file.write('y_true\ty_pred\ty_pred_calib\n')
        for true_score, predicted_score, predicted_score_calibration in zip(true_scores, predicted_scores, predicted_scores_calibration):
            the_file.write('%s\t%s\t%s\n' % (true_score, predicted_score[0], predicted_score_calibration))



def evaluate_regressor_on_samples(target_data, y_target_target_data, dirty_data, label_function, how_often, model,
                                  meta_regressor, scoring, scoring_name, dataset_name, perturbations_name, learner_name,
                                  experiment_name):


    num_examples = 0

    predicted_scores = []
    true_scores = []

    print("Evaluating on perturbed test data...")
    for _ in range(0, how_often):

        # insult_samples = target_data[target_data.label == 1]
        # noinsult_samples = target_data[target_data.label == 0]
        # sample_size = np.random.randint(0, len(insult_samples))
        # dirty_sample = dirty_data.sample(n=sample_size)
        # target_data_sample = insult_samples.sample(n=len(insult_samples) - sample_size)
        # corrupted_target_data = pd.concat([noinsult_samples, target_data_sample, dirty_sample], sort=False)

        sample_size = np.random.randint(0, len(target_data))
        target_data_sample = target_data.sample(n=len(target_data) - sample_size)
        dirty_sample = dirty_data.sample(n=sample_size)
        corrupted_target_data = pd.concat([target_data_sample, dirty_sample], sort=False)
        y_target = label_function(corrupted_target_data)


        try:
            predictions = model.predict_proba(corrupted_target_data)
            features = compute_features(predictions)
        except AttributeError:
            print("No predict_proba...")
            predictions = model.predict(corrupted_target_data)
            features = compute_regression_features(predictions)

        score_on_corrupted_target_data = scoring(y_target, model.predict(corrupted_target_data))
        predicted_score_on_corrupted_target_data = meta_regressor.predict([features])

        predicted_scores.append(predicted_score_on_corrupted_target_data)
        true_scores.append(score_on_corrupted_target_data)

        num_examples += 1
        if num_examples % 10 == 0:
            print(num_examples, '/', how_often)

    plt.plot([0, 1], [0, 1], '-', color='grey', alpha=0.5)

    min_score = np.min(predicted_scores + true_scores) - 0.05
    max_score = np.max(predicted_scores + true_scores) + 0.05

    plt.scatter(true_scores, predicted_scores, alpha=0.05)

    plt.xlabel("true " + scoring_name, fontsize=18)
    plt.ylabel("predicted " + scoring_name, fontsize=18)

    plt.xlim((min_score, max_score))
    plt.ylim((min_score, max_score))

    from sklearn.metrics import mean_squared_error, mean_absolute_error

    mse = mean_squared_error(true_scores, predicted_scores)
    mae = mean_absolute_error(true_scores, predicted_scores)

    text_x = min_score + ((max_score - min_score) / 3.0)
    text_y = min_score + ((max_score - min_score) / 10.0)

    plt.text(text_x, text_y, "MSE %.5f   MAE %.4f" % (mse, mae), fontsize=12,
             bbox=dict(facecolor='none', edgecolor='black', pad=10.0))

    print("MSE %.5f, MAE %.4f" % (mse, mae))

    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tick_params(axis='both', which='minor', labelsize=16)
    plt.legend(['perfect', 'predicted'], fontsize=18)
    plt.gcf().set_size_inches(6, 5)

    experiment_identifier = "__".join([experiment_name, dataset_name, perturbations_name, learner_name, scoring_name])

    plot_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             '../artifacts/figures/' + experiment_identifier + '.pdf')

    print("Writing plot to " + plot_file)
    plt.tight_layout()
    plt.gcf().savefig(plot_file, dpi=300)

    plt.clf()
    plt.cla()
    plt.close()

    results_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                '../artifacts/results/' + experiment_identifier + ".tsv")
    print(results_file)

    with open(results_file, 'w') as the_file:
        the_file.write('y_true\ty_pred\n')
        for true_score, predicted_score in zip(true_scores, predicted_scores):
            the_file.write('%s\t%s\n' % (true_score, predicted_score[0]))

    return mse, mae, plot_file
