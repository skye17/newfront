import os
import sys
import timeit
import numpy as np
import pandas as pd
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams
import pickle, dill
from utils import preprocess_data_for_model
from sklearn.preprocessing import StandardScaler
try:
    import PIL.Image as Image
except ImportError:
    import Image
import matplotlib.pyplot as plt
from zipline.api import order, record, symbol
from zipline.api import get_datetime, order_target
from DBN import DBN
N_MONTHS = 13


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


def initialize(context):
    pass


def load(universe, context):
    pickled = load_model(universe.load_file)
    context.model, context.scaler = pickled

    context.symbols = universe.symbols
    context.N_STOCKS = len(context.symbols)
    context.cur_time = 0
    context.day_of_month = 0
    context.month = -1
    context.count_month = -1
    context.month_sizes = [0]
    context.returns_history = np.zeros((context.N_STOCKS, 1))
    context.previous_prices = np.zeros((context.N_STOCKS, 1))
    context.previous_prices.fill(1.)


def handle_data(context, data):
    context.cur_time += 1
    month = get_datetime().date().month
    is_january = (month == 1)

    new_prices = np.array([data[symbol(symbol_name)].price for symbol_name in context.symbols], dtype='float32')
    record(Prices=new_prices)
    new_prices = new_prices.reshape((context.N_STOCKS, 1))
    #     print context.returns_history.shape
    #     print new_prices.shape
    #     print context.previous_prices.shape
    context.returns_history = np.concatenate([context.returns_history, new_prices / context.previous_prices], axis=1)
    context.previous_prices = new_prices

    if context.month != month:
        # Trading in the beginning of month
        context.month_sizes.append(context.day_of_month)
        context.day_of_month = 1
        context.count_month += 1
        context.month_sizes.append(context.day_of_month)
        context.day_of_month = 1
        if context.count_month > N_MONTHS:
            # Deleting too old returns
            if context.count_month > N_MONTHS + 1:
                context.returns_history = np.delete(context.returns_history, range(context.month_sizes[-14]), axis=1)

            model_input = preprocess_data_for_model(context.returns_history, context.month_sizes[-13:], context.scaler)
            is_january_column = np.array([is_january] * context.N_STOCKS).reshape((context.N_STOCKS, 1))
            model_input = np.concatenate([is_january_column, model_input], axis=1)
            #             print 'Input shape', model_input.shape
            predicted_proba = context.model.predict_proba(model_input)
            #             print predicted_proba

            '''
            half_trade = len(context.symbols) * 1 / 10
            args_sorted = np.argsort(predicted_proba[:, 0])
            buy_args = args_sorted[:half_trade]
            sell_args = args_sorted[-half_trade:]

            for arg in buy_args:
                order_target(symbol(context.symbols[arg]), 1)
            for arg in sell_args:
                order_target(symbol(context.symbols[arg]), -1)
            '''
            for i in range(context.N_STOCKS):
                if predicted_proba[i, 0] > 0.5:
                    order_target(symbol(context.symbols[i]), 1)
                else:
                    order_target(symbol(context.symbols[i]), -1)
    else:
        context.day_of_month += 1

    context.month = month


def analyze(results, universe):
    # Standard part
    f, (ax1, _) = plt.subplots(1, 2)
    ax1 = plt.subplot(211)
    results.portfolio_value[250:].plot(ax=ax1)
    #results.portfolio_value[400:].plot(ax=ax1)
    ax1.set_ylabel('Portfolio value (USD)')
    # Show the plot.
    plt.gcf().set_size_inches(18, 8)
    # plt.legend()
    return f
