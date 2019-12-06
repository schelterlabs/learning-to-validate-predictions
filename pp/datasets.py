import pandas as pd
import numpy as np
import os
from keras.datasets import mnist, fashion_mnist

# line	page	week	num_likes	domain	outlet	title	description	contenttype	image	url	text	id	right_of_center

DATASETS_CATEGORICAL_COLUMNS = {
    'taxi': ['VendorID', 'store_and_fwd_flag', 'RatecodeID', 'PULocationID', 'DOLocationID'],
    'articles': ['domain', 'outlet', 'contenttype'],
    'adult': ['workclass', 'occupation', 'marital_status', 'education'],
    'adult_minimal': ['workclass'],
    'heart': ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active'],
    'bank': ['job', 'marital', 'education', 'housing', 'loan', 'contact', 'default', 'campaign', 'poutcome'],
    'trolling': [],
    'housing': ['CHAS', 'RAD'],
    'bikes': ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', ]
}

DATASETS_NUMERICAL_COLUMNS = {
    'taxi': ['trip_time', 'passenger_count', 'trip_distance', 'tolls_amount', 'total_amount'],
    'articles': ['num_likes'],
    'adult': ['hours_per_week', 'age'],
    'adult_minimal': ['hours_per_week', 'age'],
    'heart': ['age_in_years', 'ap_hi', 'ap_lo', 'bmi'],
    'bank': ['balance', 'age', 'duration'],
    'trolling': [],
    'housing': ['LSTAT', 'RM'],
    'bikes': ['temp', 'atemp', 'hum', 'windspeed', 'casual']
}

DATASETS_TEXTUAL_COLUMNS = {
    'taxi': [],
    'articles': ['title'],
    'adult': [],
    'adult_minimal': [],
    'heart': [],
    'bank': [],
    'trolling': ['content'],
    'housing': [],
    'bikes': []
}


class BalancedTaxiDataset:

    # https://www1.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf
    def __init__(self):
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 '../datasets/taxi-nyc/green_tripdata_2019-01.csv')
        green = pd.read_csv(self.path)

        dropoff_time = pd.to_datetime(green.lpep_dropoff_datetime)
        pickup_time = pd.to_datetime(green.lpep_pickup_datetime)

        green['trip_time'] = (dropoff_time - pickup_time).astype('timedelta64[m]')

        disputes = green[(green.payment_type == 4)]
        no_dispute_cash = green[(green.payment_type == 2)].sample(n=int(len(disputes)/2))
        no_dispute_credit = green[(green.payment_type == 1)].sample(n=int(len(disputes)/2))

        self.categorical_columns = DATASETS_CATEGORICAL_COLUMNS[self.name()]
        self.numerical_columns = DATASETS_NUMERICAL_COLUMNS[self.name()]
        self.textual_columns = DATASETS_TEXTUAL_COLUMNS[self.name()]
        self.df = pd.concat([disputes, no_dispute_cash, no_dispute_credit])

    @staticmethod
    def name():
        return "taxi"

    @staticmethod
    def labels_from(dataframe):
        return np.array(dataframe.payment_type == 4)


class BalancedCrawledArticlesDataset:

    def __init__(self):
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 '../datasets/fb-posts/clean-articles-with-id-and-label.tsv')
        complete_data = pd.read_csv(self.path, sep='\t', encoding='latin1')

        not_right = complete_data[complete_data['right_of_center'] == False]
        right = complete_data[complete_data['right_of_center'] == True].sample(len(not_right))
        self.categorical_columns = DATASETS_CATEGORICAL_COLUMNS[self.name()]
        self.numerical_columns = DATASETS_NUMERICAL_COLUMNS[self.name()]
        self.textual_columns = DATASETS_TEXTUAL_COLUMNS[self.name()]
        self.df = pd.concat([right, not_right])
        self.df = self.df.fillna('__unknown')

    @staticmethod
    def name():
        return "articles"

    @staticmethod
    def labels_from(dataframe):
        return np.array(dataframe['right_of_center'] == True)


class BikesDataset:
    def __init__(self):

        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../datasets/bikesharing/hour.csv')
        self.df = pd.read_csv(self.path, sep=',')
        self.categorical_columns = DATASETS_CATEGORICAL_COLUMNS[self.name()]
        self.numerical_columns = DATASETS_NUMERICAL_COLUMNS[self.name()]
        self.textual_columns = DATASETS_TEXTUAL_COLUMNS[self.name()]

    @staticmethod
    def name():
        return "bikes"

    @staticmethod
    def labels_from(dataframe):
        return np.array(dataframe['cnt'])


class HousingDataset:
    def __init__(self):
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../datasets/housing/housing.data')
        self.df = pd.read_csv(self.path, sep=',')
        self.categorical_columns = DATASETS_CATEGORICAL_COLUMNS[self.name()]
        self.numerical_columns = DATASETS_NUMERICAL_COLUMNS[self.name()]
        self.textual_columns = DATASETS_TEXTUAL_COLUMNS[self.name()]

    @staticmethod
    def name():
        return "housing"

    @staticmethod
    def labels_from(dataframe):
        return np.array(dataframe['MEDV'])


class TrollingDataset:

    def __init__(self):
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../datasets/trolls/data.tsv')
        self.df = pd.read_csv(self.path, sep='\t')
        self.categorical_columns = DATASETS_CATEGORICAL_COLUMNS[self.name()]
        self.numerical_columns = DATASETS_NUMERICAL_COLUMNS[self.name()]
        self.textual_columns = DATASETS_TEXTUAL_COLUMNS[self.name()]

    @staticmethod
    def name():
        return "trolling"

    @staticmethod
    def labels_from(dataframe):
        return np.array(dataframe.label == 1)


class BalancedTrollingDataset:

    def __init__(self):
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../datasets/trolls/data.tsv')
        complete_data = pd.read_csv(self.path, sep='\t')

        trolling = complete_data[complete_data.label == 1]
        not_trolling = complete_data[complete_data.label != 1].sample(len(trolling))

        self.categorical_columns = DATASETS_CATEGORICAL_COLUMNS[self.name()]
        self.numerical_columns = DATASETS_NUMERICAL_COLUMNS[self.name()]
        self.textual_columns = DATASETS_TEXTUAL_COLUMNS[self.name()]
        self.df = pd.concat([trolling, not_trolling])

    @staticmethod
    def name():
        return "trolling"

    @staticmethod
    def labels_from(dataframe):
        return np.array(dataframe.label == 1)


class AdultDataset:

    def __init__(self):
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../datasets/adult/adult.csv')
        self.df = pd.read_csv(self.path)
        self.categorical_columns = DATASETS_CATEGORICAL_COLUMNS[self.name()]
        self.numerical_columns = DATASETS_NUMERICAL_COLUMNS[self.name()]
        self.textual_columns = DATASETS_TEXTUAL_COLUMNS[self.name()]

    @staticmethod
    def name():
        return "adult"

    @staticmethod
    def labels_from(dataframe):
        return np.array(dataframe['class'] == '>50K')


class MinimalBalancedAdultDataset:

    def __init__(self):
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../datasets/adult/adult.csv')
        complete_data = pd.read_csv(self.path)
        rich = complete_data[complete_data['class'] == '>50K']
        not_rich = complete_data[complete_data['class'] != '>50K'].sample(len(rich))
        self.categorical_columns = DATASETS_CATEGORICAL_COLUMNS[self.name()]
        self.numerical_columns = DATASETS_NUMERICAL_COLUMNS[self.name()]
        self.textual_columns = DATASETS_TEXTUAL_COLUMNS[self.name()]
        self.df = pd.concat([rich, not_rich])

    @staticmethod
    def name():
        return "adult_minimal"

    @staticmethod
    def labels_from(dataframe):
        return np.array(dataframe['class'] == '>50K')


class BalancedAdultDataset:

    def __init__(self):
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../datasets/adult/adult.csv')
        complete_data = pd.read_csv(self.path)
        rich = complete_data[complete_data['class'] == '>50K']
        not_rich = complete_data[complete_data['class'] != '>50K'].sample(len(rich))
        self.categorical_columns = DATASETS_CATEGORICAL_COLUMNS[self.name()]
        self.numerical_columns = DATASETS_NUMERICAL_COLUMNS[self.name()]
        self.textual_columns = DATASETS_TEXTUAL_COLUMNS[self.name()]
        self.df = pd.concat([rich, not_rich])

    @staticmethod
    def name():
        return "adult"

    @staticmethod
    def labels_from(dataframe):
        return np.array(dataframe['class'] == '>50K')


class BankmarketingDataset:

    def __init__(self):
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../datasets/bankmarketing/bank-full.csv')
        self.df = pd.read_csv(self.path, sep=';')
        self.categorical_columns = DATASETS_CATEGORICAL_COLUMNS[self.name()]
        self.numerical_columns = DATASETS_NUMERICAL_COLUMNS[self.name()]
        self.textual_columns = DATASETS_TEXTUAL_COLUMNS[self.name()]

    @staticmethod
    def name():
        return "bank"

    @staticmethod
    def labels_from(dataframe):
        return np.array(dataframe.y == 'yes')


class BalancedBankmarketingDataset:

    def __init__(self):
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../datasets/bankmarketing/bank-full.csv')
        complete_data = pd.read_csv(self.path, sep=';')
        subscribed = complete_data[complete_data.y == 'yes']
        subscribed_not = complete_data[complete_data.y != 'yes'].sample(len(subscribed))

        self.categorical_columns = DATASETS_CATEGORICAL_COLUMNS[self.name()]
        self.numerical_columns = DATASETS_NUMERICAL_COLUMNS[self.name()]
        self.textual_columns = DATASETS_TEXTUAL_COLUMNS[self.name()]
        self.df = pd.concat([subscribed, subscribed_not])

    @staticmethod
    def name():
        return "bank"

    @staticmethod
    def labels_from(dataframe):
        return np.array(dataframe.y == 'yes')


class CardioDataset:

    def __init__(self):
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../datasets/cardio/cardio_train.csv')
        data = pd.read_csv(self.path, sep=';')
        data['bmi'] = data['weight'] / (.01 * data['height']) ** 2
        data['age_in_years'] = data['age'] / 365

        self.categorical_columns = DATASETS_CATEGORICAL_COLUMNS[self.name()]
        self.numerical_columns = DATASETS_NUMERICAL_COLUMNS[self.name()]
        self.textual_columns = DATASETS_TEXTUAL_COLUMNS[self.name()]
        self.df = data

    @staticmethod
    def name():
        return "heart"

    @staticmethod
    def labels_from(dataframe):
        return np.array(dataframe.cardio == 1)


class MnistDataset:
    def __init__(self):
        img_rows, img_cols = 28, 28
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        class_a = 3
        class_b = 5

        train_a_indexes = np.where(y_train == class_a)[0]
        x_train_a = x_train[train_a_indexes, ]
        y_train_a = y_train[train_a_indexes]

        train_b_indexes = np.where(y_train == class_b)[0]
        x_train_b = x_train[train_b_indexes, ]
        y_train_b = y_train[train_b_indexes]

        test_a_indexes = np.where(y_test == class_a)[0]
        x_test_a = x_test[test_a_indexes, ]
        y_test_a = y_test[test_a_indexes]

        test_b_indexes = np.where(y_test == class_b)[0]
        x_test_b = x_test[test_b_indexes, ]
        y_test_b = y_test[test_b_indexes]

        X = np.vstack((x_train_a, x_train_b, x_test_a, x_test_b))
        y = np.hstack((y_train_a, y_train_b, y_test_a, y_test_b))

        y[y == class_a] = 0
        y[y == class_b] = 1

        data = pd.DataFrame()
        data['image'] = X.reshape((X.shape[0], 28 * 28)).tolist()
        data['label'] = y.tolist()

        self.categorical_columns = []
        self.numerical_columns = []
        self.textual_columns = []
        self.df = data

    @staticmethod
    def name():
        return "mnist"

    @staticmethod
    def labels_from(dataframe):
        return dataframe.label


class FashionDataset:
    def __init__(self):
        img_rows, img_cols = 28, 28
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        class_a = 7
        class_b = 9

        train_a_indexes = np.where(y_train == class_a)[0]
        x_train_a = x_train[train_a_indexes, ]
        y_train_a = y_train[train_a_indexes]

        train_b_indexes = np.where(y_train == class_b)[0]
        x_train_b = x_train[train_b_indexes, ]
        y_train_b = y_train[train_b_indexes]

        test_a_indexes = np.where(y_test == class_a)[0]
        x_test_a = x_test[test_a_indexes, ]
        y_test_a = y_test[test_a_indexes]

        test_b_indexes = np.where(y_test == class_b)[0]
        x_test_b = x_test[test_b_indexes, ]
        y_test_b = y_test[test_b_indexes]

        X = np.vstack((x_train_a, x_train_b, x_test_a, x_test_b))
        y = np.hstack((y_train_a, y_train_b, y_test_a, y_test_b))

        y[y == class_a] = 0
        y[y == class_b] = 1

        data = pd.DataFrame()
        data['image'] = X.reshape((X.shape[0], 28 * 28)).tolist()
        data['label'] = y.tolist()

        self.categorical_columns = []
        self.numerical_columns = []
        self.textual_columns = []
        self.df = data

    @staticmethod
    def name():
        return "fashion"

    @staticmethod
    def labels_from(dataframe):
        return dataframe.label
