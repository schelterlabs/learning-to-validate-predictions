import random
import numpy as np
import cv2


class Mixture:
    def __init__(self, perturbations):
        self.perturbations = perturbations

    def transform(self, clean_df):
        df = clean_df
        for perturbation in self.perturbations:
            df = perturbation.transform(df)
        return df


class MissingValues:

    def __init__(self, fraction, columns, value_to_put_in):
        self.fraction = fraction
        self.columns = columns
        self.value_to_put_in = value_to_put_in

    def transform(self, clean_df):
        # we operate on a copy of the data
        df = clean_df.copy(deep=True)

        row_indexes = [row for row in range(df.shape[0])]
        num_rows_to_pick = int(round(self.fraction * df.shape[0]))
        affected_indexes = set(random.sample(row_indexes, num_rows_to_pick))
        row_index_indicators = [row in affected_indexes for row in range(df.shape[0])]

        df.loc[row_index_indicators, self.columns] = self.value_to_put_in

        return df


class LassoExperiment:

    def __init__(self, column, ignored_values):
        self.column = column
        self.ignored_values = ignored_values

    def transform(self, clean_df):
        # we operate on a copy of the data
        df = clean_df.copy(deep=True)

        df.loc[df['workclass'].isin(self.ignored_values), 'workclass'] = 'some_random_string'

        return df


class Typo:
    def __init__(self, fraction, columns):
        self.fraction = fraction
        self.columns = columns

    def shuffle_word(self, word):
        word = list(str(word))
        random.shuffle(word)
        return ''.join(word)

    def transform(self, clean_df):
        # print(clean_df.dtypes)
        # we operate on a copy of the data
        df = clean_df.copy(deep=True)
        for index, row in df.iterrows():
            for column in self.columns:
                if random.random() < self.fraction:
                    shuffled = self.shuffle_word(row[column])
                    # print(column, df[column].dtype, row[column], shuffled)

                    if df[column].dtype == np.int64:
                        df.at[index, column] = round(float(shuffled))
                    else:
                        df.at[index, column] = shuffled

        return df


class FlipSign:
    def __init__(self, fraction, columns):
        self.fraction = fraction
        self.columns = columns

    def transform(self, clean_df):
        # we operate on a copy of the data
        df = clean_df.copy(deep=True)

        for index, row in df.iterrows():
            for column in self.columns:
                if random.random() < self.fraction:
                    df.at[index, column] = row[column] * -1

        return df


class PlusMinusSomePercent:
    def __init__(self, fraction, columns):
        self.fraction = fraction
        self.columns = columns

    def transform(self, clean_df):
        # we operate on a copy of the data
        df = clean_df.copy(deep=True)

        for index, row in df.iterrows():
            for column in self.columns:
                if random.random() < self.fraction:
                    df.at[index, column] = row[column] + row[column] * np.random.uniform(-0.1, 0.1)

        return df



class Leetspeak:

    def __init__(self, fraction, column, label_column, label_value):
        self.fraction = fraction
        self.column = column
        self.label_column = label_column
        self.label_value = label_value

    def transform(self, clean_df):
        # we operate on a copy of the data
        df = clean_df.copy(deep=True)

        sampled_df = df[df[self.label_column] == self.label_value]
        for index, row in sampled_df.iterrows():

            if random.random() < self.fraction:
                leet_content = row[self.column] \
                    .replace('a', '4') \
                    .replace('e', '3') \
                    .replace('l', '1') \
                    .replace('t', '7') \
                    .replace('s', '5') \
                    .replace('o', '0')
                df.at[index, self.column] = leet_content

        return df


class Outliers:

    def __init__(self, fraction, columns):
        self.fraction = fraction
        self.columns = columns

    def transform(self, clean_df):
        # we operate on a copy of the data
        df = clean_df.copy(deep=True)
        stddevs = {column: np.std(df[column]) for column in self.columns}
        scales = {column: random.uniform(1, 5) for column in self.columns}

        for index, row in df.iterrows():

            for column in self.columns:
                if random.random() < self.fraction:
                    noise = np.random.normal(0, scales[column] * stddevs[column])
                    outlier = df.at[index, column] + noise

                    df.at[index, column] = outlier

        return df


class Scaling:

    def __init__(self, fraction, columns):
        self.fraction = fraction
        self.columns = columns

    def transform(self, clean_df):
        # we operate on a copy of the data
        df = clean_df.copy(deep=True)

        scale_factor = np.random.choice([10, 100, 1000])

        for index, row in df.iterrows():
            for column in self.columns:
                if random.random() < self.fraction:
                    df.at[index, column] = df.at[index, column] * scale_factor

        return df


class SwappedValues:

    def __init__(self, fraction, column_pair):
        self.fraction = fraction
        self.column_pair = column_pair

    def transform(self, clean_df):
        # we operate on a copy of the data
        df = clean_df.copy(deep=True)

        (column_a, column_b) = self.column_pair

        values_of_column_a = list(df[column_a])
        values_of_column_b = list(df[column_b])

        for index in range(0, len(values_of_column_a)):
            if random.random() < self.fraction:
                temp_value = values_of_column_a[index]
                values_of_column_a[index] = values_of_column_b[index]
                values_of_column_b[index] = temp_value

        df[column_a] = values_of_column_a
        df[column_b] = values_of_column_b

        return df


class ImageRotation:

    def __init__(self, fraction):
        self.fraction = fraction

    def transform(self, clean_df):
        # we operate on a copy of the data
        df = clean_df.copy(deep=True)

        serialized_images = list(df['image'])

        for index in range(0, len(serialized_images)):
            if random.random() < self.fraction:
                raw_image = serialized_images[index]

                img = np.zeros([28, 28, 3])
                img[:, :, 0] = raw_image.reshape(28, 28)
                img[:, :, 1] = raw_image.reshape(28, 28)
                img[:, :, 2] = raw_image.reshape(28, 28)

                degree = np.random.randint(0, 359)

                rotation = cv2.getRotationMatrix2D((14, 14), degree, 1)
                rotated = cv2.warpAffine(img, rotation, (28, 28))
                serialized_images[index] = rotated[:, :, 0].reshape(784,)

        df['image'] = serialized_images

        return df


class NoisyImage:

    def __init__(self, fraction):
        self.fraction = fraction

    def transform(self, clean_df):
        # we operate on a copy of the data
        df = clean_df.copy(deep=True)

        serialized_images = list(df['image'])

        for index in range(0, len(serialized_images)):
            if random.random() < self.fraction:
                raw_image = serialized_images[index]

                img = np.zeros([28, 28, 3])
                img[:, :, 0] = raw_image.reshape(28, 28)
                img[:, :, 1] = raw_image.reshape(28, 28)
                img[:, :, 2] = raw_image.reshape(28, 28)

                mean = 0
                var = np.random.uniform(0.05, 0.5)
                sigma = var ** 0.5
                gaussian = np.random.normal(mean, sigma, (28, 28))
                noisy_image = np.zeros(img.shape, np.float32)

                if len(img.shape) == 2:
                    noisy_image = img + gaussian
                else:
                    noisy_image[:, :, 0] = img[:, :, 0] + gaussian
                    noisy_image[:, :, 1] = img[:, :, 1] + gaussian
                    noisy_image[:, :, 2] = img[:, :, 2] + gaussian

                cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
                noisy_image = noisy_image.astype(np.uint8)

                serialized_images[index] = noisy_image[:, :, 0].reshape(784,)

        df['image'] = serialized_images

        return df
