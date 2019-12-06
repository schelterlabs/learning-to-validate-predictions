import numpy as np
from pp import perturbations


def missing_perturbations(columns, substitute, repetitions, probs=[0.0, 0.05, 0.25, 0.5, 0.75, 0.99]):
    _perturbations = []
    for num_columns_affected in range(1, len(columns) + 1):
        for fraction_of_values_to_delete in probs:
            for _ in range(0, repetitions):
                columns_affected = np.random.choice(columns, num_columns_affected)
                _perturbations.append(perturbations.MissingValues(fraction_of_values_to_delete,
                                      columns_affected, substitute))

    return _perturbations


def outlier_perturbations(_numerical_columns, repetitions, probs=[0.0, 0.05, 0.25, 0.5, 0.75, 0.99]):
    _perturbations = []
    for num_columns_affected in range(1, len(_numerical_columns) + 1):
        for fraction_of_outliers in probs:
            for _ in range(0, repetitions):
                columns_affected = np.random.choice(_numerical_columns, num_columns_affected)
                _perturbations.append(perturbations.Outliers(fraction_of_outliers, columns_affected))

    return _perturbations


def swapped_perturbations(_affected_column_pairs, repetitions, probs=[0.0, 0.05, 0.25, 0.5, 0.75, 0.99]):
    _perturbations = []
    for column_pair in _affected_column_pairs:
        for fraction_of_outliers in probs:
            for _ in range(0, repetitions):
                _perturbations.append(perturbations.SwappedValues(fraction_of_outliers, column_pair))

    return _perturbations


def scaling_perturbations(_numerical_columns, repetitions, probs=[0.0, 0.05, 0.25, 0.5, 0.75, 0.99]):
    _perturbations = []
    for num_columns_affected in range(1, len(_numerical_columns) + 1):
        for fraction_of_outliers in probs:
            for _ in range(0, repetitions):
                columns_affected = np.random.choice(_numerical_columns, num_columns_affected)
                _perturbations.append(perturbations.Scaling(fraction_of_outliers, columns_affected))

    return _perturbations


def flipsign_perturbations(_numerical_columns, repetitions, probs=[0.0, 0.05, 0.25, 0.5, 0.75, 0.99]):
    _perturbations = []
    for num_columns_affected in range(1, len(_numerical_columns) + 1):
        for prob in probs:
            for _ in range(0, repetitions):
                columns_affected = np.random.choice(_numerical_columns, num_columns_affected)
                _perturbations.append(perturbations.FlipSign(prob, columns_affected))

    return _perturbations


def plusminus_perturbations(_numerical_columns, repetitions, probs=[0.0, 0.05, 0.25, 0.5, 0.75, 0.99]):
    _perturbations = []
    for num_columns_affected in range(1, len(_numerical_columns) + 1):
        for prob in probs:
            for _ in range(0, repetitions):
                columns_affected = np.random.choice(_numerical_columns, num_columns_affected)
                _perturbations.append(perturbations.PlusMinusSomePercent(prob, columns_affected))

    return _perturbations


def typo_perturbations(_categorical_columns, repetitions, probs=[0.0, 0.05, 0.25, 0.5, 0.75, 0.99]):
    _perturbations = []
    for num_columns_affected in range(1, len(_categorical_columns) + 1):
        for fraction_of_values in probs:
            for _ in range(0, repetitions):
                columns_affected = np.random.choice(_categorical_columns, num_columns_affected)
                _perturbations.append(perturbations.Typo(fraction_of_values, columns_affected))

    return _perturbations
