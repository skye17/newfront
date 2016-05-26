import argparse
import os
import utils

import numpy as np
import pandas as pd


def check_from_zero_to_one(value):
    """
    Checker for ArgumentParser.

    :param value:
    :return:
    """
    fvalue = float(value)
    if fvalue <= 0.0 or fvalue > 1.0:
        raise argparse.ArgumentTypeError("%s is an invalid float value "
                                         "from zero to one" % value)
    return fvalue


parser = argparse.ArgumentParser(description='Script for preprocessing data '
                                             'for DBN-MLP models.')

parser.add_argument('--input', type=str, default='.',
                    help='Directory with stocks to train on.')
parser.add_argument('--output', default='./output.csv', type=str,
                    help='Directory where to put output files. Default '
                         '\'./output.csv\'')

parser.add_argument('--train_size', default=0.7, type=check_from_zero_to_one,
                    help='Proportion of the dataset to '
                         'include in the train split. Default 0.7')

parser.add_argument('--date_col', default='Date', type=str,
                    help='Name of the columns which parse as date. Default '
                         '\'Date\'')

parser.add_argument('--open_col', default='Open', type=str,
                    help='Name of the columns which parse as open price. '
                         'Default \'Open\'')

parser.add_argument('--close_col', default='Close', type=str,
                    help='Name of the columns which parse as close price. '
                         'Default \'Close\'')

parser.add_argument('--sep', default=',', type=str,
                    help='Separator in input files. Default \',\'')

parser.add_argument('--with_target_returns', default=False, type=bool,
                    help='This flag specifies whether to return target '
                         'month returns or not. Default False.')

parser.add_argument('--log_returns', default=False, type=bool,
                    help='This flag specifies whether to use log-returns. '
                         'Default False.')

namespace = None
train_size = date_col = open_col = close_col = \
    sep = with_target_returns = log_returns = None

n_days_to_retain = 18
day_cols = ['DAY{0}'.format(day) for day in
            xrange(1, n_days_to_retain + 1)]
month_cols = ['MONTH{0}'.format(m) for m in xrange(1, 13)]

day_month_cols = day_cols + month_cols


def init_namespace_and_vars():
    global namespace, train_size, n_days_to_retain, date_col, \
        open_col, close_col, sep, with_target_returns, log_returns
    namespace = parser.parse_args()
    train_size = namespace.train_size
    date_col = namespace.date_col
    open_col = namespace.open_col
    close_col = namespace.close_col
    sep = namespace.sep
    with_target_returns = namespace.with_target_returns
    log_returns = namespace.log_returns


def get_stocks_months_returns(dataframes):
    """
    This function returns dataframe that contains monthly returns for each (
    stock, month) pair. This dataframe is needed for labeling data.

    :param dataframes: list (for each stock) of dataframes with columns
    'MONTH' and 'MONTHRETURN'
    :return: pd.DataFrame
    """
    if len(dataframes) == 0:
        return None
    whole_df = dataframes[0].drop_duplicates('MONTH')[['MONTH', 'MONTHRETURN']]
    for i in xrange(1, len(dataframes)):
        df = dataframes[i].drop_duplicates('MONTH')[['MONTH', 'MONTHRETURN']]
        whole_df = pd.concat((whole_df, df), ignore_index=True)
    whole_df.dropna(inplace=True)
    return whole_df


def read_dataset_and_make_features(path):
    try:
        df = pd.read_csv(path, sep=sep, parse_dates=[date_col],
                         usecols=[date_col, open_col, close_col])
    except ValueError:
        return None
    if df.shape[0] < 390:  # min ~13 months are needed.
        return None
    df.dropna(inplace=True)
    df.sort_values(by=date_col, inplace=True)
    df['DAY'] = df[date_col].dt.day
    df['MONTH'] = (df[date_col].dt.year - 1970) * 12 + df[date_col].dt.month
    df['DAYRETURN'] = df[close_col] / df[open_col]
    df['ISJAN'] = (df[date_col].dt.month == 1)
    df['ISJAN'] = df['ISJAN'].astype(float)
    df.drop(date_col, axis=1, inplace=True)
    months = df['MONTH'].drop_duplicates()
    # Make features.
    idx_to_drop = []
    # Drop months which don't have enough data.
    for m in months:
        mask = df['MONTH'] == m
        idx = df.loc[mask].index.tolist()
        if len(idx) < n_days_to_retain:
            idx_to_drop.extend(idx)
        else:
            # Retain last 18 days, drop others.
            idx_to_drop.extend(idx[:len(idx) - n_days_to_retain])
    df.drop(idx_to_drop, axis=0, inplace=True)
    df['DAYCUMRETURN'] = 0.0
    df['MONTHRETURN'] = 0.0
    # Calculating day cumulative returns and monthly returns.
    months = df['MONTH'].drop_duplicates()  # Maybe we dropped
    # some months at all.
    for m in months:
        mask = df['MONTH'] == m
        month_open = df.loc[mask, open_col].iloc[0]
        month_close = df.loc[mask, close_col].iloc[-1]
        month_return = month_close / month_open
        day_returns = df[mask]['DAYRETURN']
        day_cum_returns = np.zeros(day_returns.shape[0])
        day_cum_returns[0] = day_returns.iloc[0]
        for i in xrange(1, day_returns.shape[0]):
            day_cum_returns[i] = day_cum_returns[i - 1] * day_returns.iloc[i]
        df.loc[mask, ['DAYCUMRETURN']] = day_cum_returns
        df.loc[mask, ['MONTHRETURN']] = month_return
    df.drop(['DAYRETURN'], axis=1, inplace=True)
    df['STOCKNAME'] = utils.get_fname(path).split('.')[0]
    if df.isnull().sum().sum() > 0:
        return None
    return df


def make_final_stock_dataset(df, stock2month_returns):
    assert df.shape[0] > 0
    assert stock2month_returns is not None
    global log_returns, with_target_returns

    def f(x):
        if log_returns:
            return np.log(x)
        else:
            return x

    day_csums_df = pd.DataFrame()
    months = df['MONTH'].drop_duplicates()
    groupby = df.groupby('MONTH')['DAYCUMRETURN']
    monthly_returns = df.groupby('MONTH')['MONTHRETURN'].first()
    assert len(months) == len(monthly_returns)
    for j in xrange(13, months.size - 1):
        mask_nextmonth = stock2month_returns['MONTH'] == j + 1
        can_label = (mask_nextmonth.shape[0] > 0)
        if not can_label:
            continue
        m = months.iloc[j]
        # Generating n_days_to_retain features with cumulative day returns.
        isjan = df[df['MONTH'] == m]['ISJAN'].iloc[0]
        row_m_isjan_days = pd.concat(
            (pd.Series([m, isjan]), groupby.get_group(m)),
            ignore_index=True)
        # Generating 12 montly returns as features.
        row_months = np.array(monthly_returns.iloc[j - 13:j - 1])
        for i in xrange(1, row_months.shape[0]):
            row_months[i] = row_months[i - 1] * row_months[i]
        row_months = pd.Series(row_months)
        target_month_returns = monthly_returns.iloc[j + 1]
        # Determining label for month t:
        # Compare t+1 month returns with median for
        # all stocks' returns in month t+1
        # median_all_stocks = np.median(
            # stock2month_returns[mask_nextmonth]['MONTHRETURN'])
        label = (target_month_returns > np.median(row_months.values))
        if with_target_returns:
            row = pd.concat((row_m_isjan_days, row_months,
                             pd.Series([label, target_month_returns])),
                            ignore_index=True)
        else:
            row = pd.concat((row_m_isjan_days, row_months, pd.Series([label])),
                            ignore_index=True)
        day_csums_df = day_csums_df.append(row, ignore_index=True)
    if with_target_returns:
        day_csums_df.columns = ['MONTH', 'ISJAN'] + day_cols + \
                               month_cols + ['LABEL', 'TARGETRETURNS']
    else:
        day_csums_df.columns = ['MONTH', 'ISJAN'] + day_month_cols + \
                               ['LABEL']
    day_csums_df['STOCKNAME'] = df['STOCKNAME'].iloc[0]  # all stocknames are
    # equal within `df` dataframe
    day_csums_df[day_month_cols] = f(day_csums_df[day_month_cols])
    day_csums_df.dropna(inplace=True)
    return day_csums_df


if __name__ == '__main__':
    init_namespace_and_vars()
    assert not os.path.isdir(namespace.output)
    filenames = [utils.get_path(fname, namespace.input) for fname
                 in os.listdir(namespace.input)]

    print "Reading datasets for each stock and making features for them..."
    n_files = len(filenames)
    print "Total number of files:", n_files
    dfs = []
    for i in xrange(n_files):
        full_fname = filenames[i]
        df = read_dataset_and_make_features(full_fname)
        if df is not None:
            dfs.append(df)
        if (i + 1) % 100 == 0:
            print "{0} -- {1:.2f}%".format(i + 1, float(i + 1) / n_files * 100)

    print "Making stock-month returns dataset for labeling data in future."
    stock2month_returns = get_stocks_months_returns(dfs)
    print "NaNs in stock2month", stock2month_returns.isnull().sum().sum()

    print "Making final dataset for each stock..."
    n_dfs = len(dfs)
    final_dfs = []
    for i in xrange(n_dfs):
        df = dfs[i]
        final_df = make_final_stock_dataset(df, stock2month_returns)
        final_dfs.append(final_df)
        if (i + 1) % 100 == 0:
            print "{0} -- {1:.2f}%".format(i + 1, float(i + 1) / n_dfs * 100)

    print "Merging datasets and sorting..."
    # Build all train dataset
    merged = utils.merge_dataframes(final_dfs)
    # Sort by time
    merged = utils.sort_by_month(merged, month_col='MONTH', drop=False)
    utils.drop_inf(merged)
    # Normalize dataset
    print "NaNs before normalizing dataset", merged.isnull().sum().sum()
    merged_scaled, scaler = utils.normalize_dataset(merged[day_month_cols])
    merged[day_month_cols] = merged_scaled
    print "NaNs after normalizing dataset", merged.isnull().sum().sum()

    if train_size == 1.0:
        train, test = merged, None
    else:
        bound = int(train_size * merged.shape[0])
        train, test = merged[:bound], merged[bound:]

    dirname = utils.get_dirname(namespace.output)
    out_fname = utils.get_fname(namespace.output).split('.')[0]
    train_fname = utils.get_path(out_fname + '_train.csv', dirname)
    test_fname = utils.get_path(out_fname + '_test.csv', dirname)
    scaler_fname = utils.get_path(out_fname + '_scaler', dirname)
    print "Writing train to %s" % train_fname
    train.to_csv(train_fname, index=False)
    if test is not None:
        print "Writing test to %s" % test_fname
        test.to_csv(test_fname, index=False)

    print "Dumping scaler to %s" % scaler_fname
    utils.dump_scaler(scaler, scaler_fname)
    print "Done."