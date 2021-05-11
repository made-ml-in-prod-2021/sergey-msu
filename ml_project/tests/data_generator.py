import random
import numpy as np
import pandas as pd


def generate_test_dataframe(data_len, random_state, cols_data=None):
    """ Generates random tests data. """
    random.seed(random_state)
    np.random.seed(random_state)

    data = pd.DataFrame({
        'age': np.random.randint(20, 81, data_len),
        'cp': np.random.randint(0, 4, data_len),
        'sex': np.random.randint(0, 2, data_len),
        'trestbps': np.random.randint(100, 181, data_len),
        'chol': np.random.randint(150, 380, data_len),
        'fbs': np.random.randint(0, 2, data_len),
        'restecg': np.random.randint(0, 2, data_len),
        'thalach': np.random.randint(120, 211, data_len),
        'exang': np.random.randint(0, 2, data_len),
        'oldpeak': 4*np.random.rand(data_len),
        'slope': np.random.randint(0, 3, data_len),
        'ca': np.random.randint(0, 5, data_len),
        'thal': np.random.randint(1, 4, data_len),
        'target': np.random.randint(0, 2, data_len),
    })

    if cols_data:
        for col in cols_data:
            data[col] = cols_data[col]

    return data
