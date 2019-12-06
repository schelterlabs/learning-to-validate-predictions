import warnings
# needed to get rid of deprecation warnings
import imp
from joblib import dump
from pp import datasets
from pp import learners
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

#dataset = datasets.BankmarketingDataset()
#learner = learners.DNN('roc_auc')
#model_directory = 'black-box-models/dnn-bank-roc_auc/'

#dataset = datasets.HousingDataset()
#learner = learners.RidgeRegression('neg_mean_absolute_error')
#model_directory = 'black-box-models/ridge-housing-mae/'


dataset = datasets.BalancedTaxiDataset()
learner = learners.AutoSklearn('accuracy', 600)
model_directory = 'black-box-models/autosklearn-taxi-accuracy/'



#dataset = datasets.MnistDataset()
#learner = learners.AutoKerasLarge('accuracy')
#model_directory = 'black-box-models/autokeraslarge-mnist-accuracy/'


#dataset = datasets.BalancedAdultDataset()
#learner = learners.TPOT('accuracy')
#model_directory = 'black-box-models/tpot-adult-accuracy/'

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


dump(model, model_directory + 'model.joblib')
dump(y_train, model_directory + 'y_train.joblib')
dump(y_test, model_directory + 'y_test.joblib')
dump(y_target, model_directory + 'y_target.joblib')

train_data.to_csv(model_directory + 'train.csv', sep='\t', encoding='utf-8')
test_data.to_csv(model_directory + 'test.csv', sep='\t', encoding='utf-8')
target_data.to_csv(model_directory + 'target.csv', sep='\t', encoding='utf-8')
