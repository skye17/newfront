import random
import lasagne
import pytz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from zipline.algorithm import TradingAlgorithm
from zipline.api import order, record, symbol, set_commission
from zipline.finance import commission


import time

import theano
import theano.tensor as T

N_ASSETS = 5
BATCH_SIZE = 64
GRANULARITY = 100
CNT_NEURONS = 128
lr = 0.01
CNT_EPOCHS = 200


def initialize():
    pass


def build_network(input_var=None):
    network = lasagne.layers.InputLayer(shape=(BATCH_SIZE, 2 * N_ASSETS), input_var=input_var)
    network = lasagne.layers.DenseLayer(network, CNT_NEURONS, nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.DenseLayer(network, CNT_NEURONS, nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.DenseLayer(network, CNT_NEURONS, nonlinearity=lasagne.nonlinearities.rectify)
    networks = []
    for i in range(N_ASSETS):
        networks.append(lasagne.layers.DenseLayer(network, 3, nonlinearity=lasagne.nonlinearities.softmax))
    network = lasagne.layers.concat(networks, axis=0)
    return network


def calc_portfolio(normed_price_diffs, context):
    input_var = np.array(list(normed_price_diffs) + list(context.portf))
    prediction = context.val_fn(input_var.reshape((1, 10))).reshape((5, 3))
    return np.argmax(prediction, axis=1) - 1


def handle_data(context, data):
    context.cnt += 1
    new_prices = np.array([data[symbol(asset_name)].price for asset_name in context.panel.axes[0]], dtype='float32')
    # print new_prices
    if context.check == 0:
        context.check = 1
        context.old_prices = new_prices
        return
    price_diffs = new_prices - context.old_prices
    normed_price_diffs = price_diffs / context.old_prices
    # print context.portfolio['portfolio_value']
    # new_portf = context.action
    # new_portf = calc_portfolio(normed_price_diffs, context)
    new_portf = context.pred[context.cnt]
    for i in np.arange(len(new_portf)):
        name = symbol(context.panel.axes[0][i])
        num = new_portf[i] - context.portf[i]
        order(name, num * 100)

    context.portf = new_portf
    context.old_prices = new_prices


def load(universe, context):
    context.input_var = T.matrix('inputs')
    context.network = build_network(context.input_var)
    context.price_diffs = context.input_var[:, N_ASSETS:]
    context.target = T.ivector("targets")

    context.predictions = lasagne.layers.get_output(context.network)
    context.loss = lasagne.objectives.categorical_crossentropy(context.predictions, context.target)
    context.loss = context.loss.mean()

    context.all_params = lasagne.layers.get_all_params(context.network, trainable=True)
    context.updates = lasagne.updates.nesterov_momentum(context.loss, context.all_params,
                                                        learning_rate=lr, momentum=0.9)

    # As a bonus, also create an expression for the classification accuracy:
    # test_acc = T.mean(T.eq(T.argmax(predictions, axis=1), target), dtype=theano.config.floatX)
    context.train_fn = theano.function([context.input_var, context.target], context.loss,
                                       updates=context.updates, allow_input_downcast=True)

    # Compile a second function computing the validation loss and accuracy:
    context.val_fn = theano.function([context.input_var], context.predictions, allow_input_downcast=True)
    context.check = 0
    context.old_prices = np.zeros(N_ASSETS)
    context.portf = np.zeros(N_ASSETS)
    set_commission(commission.PerDollar(0.0006))
    context.action = np.zeros(5)
    context.pred = pd.read_csv(universe.load_file, header=None).values
    context.cnt = 0











