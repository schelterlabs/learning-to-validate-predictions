from pp.serialization import load_black_box

# model_directory = 'lr-adult-accuracy'
# model_directory = 'lr-heart-accuracy'
# model_directory = 'dnn-trolling-roc_auc'
model_directory = 'convnet-mnist-accuracy'

(model, scoring, scoring_name, train_data, y_train, test_data, y_test, target_data, y_target, _, _) = \
    load_black_box(model_directory)


print(scoring_name, "on train set", scoring(y_train, model.predict(train_data)))
print(scoring_name, "on test set", scoring(y_test, model.predict(test_data)))
print(scoring_name, "on target set", scoring(y_target, model.predict(target_data)))
