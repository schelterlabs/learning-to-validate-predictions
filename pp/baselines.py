from scipy import stats
import numpy as np


def relational_shift(sample1, sample2, numerical_columns, categorical_columns):
    p_vals = []
    for column in numerical_columns:
        _, p_val = stats.ks_2samp(sample1[column], sample2[column])
        p_vals.append(p_val)

    for column in categorical_columns:
        counts1 = {str(key): val for key, val in dict(sample1[column].value_counts()).items()}
        counts2 = {str(key): val for key, val in dict(sample2[column].value_counts()).items()}


        all_keys = list(set(list(counts1.keys()) + list(counts2.keys())))
        all_keys.sort()
        all_keys_counts1 = []
        all_keys_counts2 = []

        for key in all_keys:
            if key in counts1:
                all_keys_counts1.append(counts1[key])
            else:
                all_keys_counts1.append(0)

            if key in counts2:
                all_keys_counts2.append(counts2[key])
            else:
                all_keys_counts2.append(0)
        _, p_val, _, _ = stats.chi2_contingency([all_keys_counts1, all_keys_counts2])
        p_vals.append(p_val)

    alpha = 0.05
    K = len(p_vals)
    return np.min(p_vals) < (alpha / K)
