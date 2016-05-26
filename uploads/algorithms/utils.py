import dill
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_data_for_model(data, month_sizes, scaler, n_trading_days=18,
                              log_returns=False):
    """
    Function to prepare `data` for predicting with DBN model.

    :param data: numpy-2d-array with a shape (n_stocks,
    n_days_to_retain*13). Each cell contain daily return.

    :param month_sizes: list (len=12) number of trading days for each month
    except last.

    :param scaler: sklearn.StandardScaler to scale array with
    `n_tradin_days`+12 features.

    :param n_trading_days: number of trading days in last month

    :param log_returns: flag specifies whether return logarithmic returns

    :returns numpy-2d-array with sample for each stock
    """
    data_num_cols = sum(month_sizes)
    n_features = 12 + n_trading_days
    res = np.zeros((data.shape[0], n_features))
    assert scaler.mean_.size == n_features
    # Reducing columns for each month to get monthly returns.
    for month in xrange(12):
        month_size = month_sizes[month]
        # Taking returns in current month.
        returns = data[:, month * month_size:(month + 1) * month_size]
        res[:, month] = np.prod(returns, axis=1)

    # Calculating cumulative products over months.
    for month in xrange(1, 12):
        res[:, month] = res[:, month - 1] * res[:, month]

    # Calculating cumulative products over days in last month.
    returns = data[:, -n_trading_days:]
    res[:, n_features-n_trading_days] = returns[:, 0]
    for day in xrange(1, n_trading_days):
        idx_to_place = n_features-n_trading_days+day
        res[:, idx_to_place] = returns[:, day] * res[:, idx_to_place - 1]
    if log_returns:
        res = np.log(res)
    res = scaler.transform(res)
    return res


def normalize_dataset(dataset):
    """
    Function to normalize dataset with StandardScaler.

    :param dataset: pd.DataFrame
    """
    columns = dataset.columns
    array = dataset.values
    scaler = StandardScaler()
    array = scaler.fit_transform(array)
    return pd.DataFrame(data=array, columns=columns), scaler


def get_path(name, prefix):
    """
    This function returns full path to specified file.

    :param name: name of dir or file
    :param prefix: Path to folder containing `name`
    :return: str
    """
    return os.path.join(prefix, name)


def get_fname(path):
    return os.path.split(os.path.abspath(path))[1]


def get_dirname(path):
    return os.path.split(os.path.abspath(path))[0]


def merge_dataframes(dataframes):
    """
    This function merges dataframes in one.

    :param dataframes: list of dataframes to merge
    :return: merged dataframe
    """
    if len(dataframes) == 0:
        return None
    else:
        merged = dataframes[0]
        for i in xrange(1, len(dataframes)):
            merged = pd.concat((merged, dataframes[i]), ignore_index=True)
        return merged


def drop_inf(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)


def sort_by_month(dataframe, month_col, drop=False):
    dataframe.sort_values(by=month_col, inplace=True)
    if drop:
        dataframe.drop(month_col, inplace=True)
    return dataframe


def dump_scaler(scaler, path):
    with open(path, 'wb') as f:
        f.write(pickle.dumps(scaler))


def load_scaler(path):
    with open(path, 'rb') as f:
        return pickle.loads(f.read())


def dump_model(model, path):
    with open(path, 'wb') as f:
        f.write(dill.dumps(model))


def load_model(path):
    with open(path, 'rb') as f:
        return dill.loads(f.read())
