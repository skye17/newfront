import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T
from zipline.algorithm import TradingAlgorithm
from zipline.api import order, record, symbol, add_history, history
import lasagne



class My_history_with_weekends(object):
    def __init__(self, length, mean, scale):
        self.queue = []
        self.norm_queue = []
        self.queue_length = length
        self.mean = mean
        self.scale = scale

    def normalize(self, price):
        return 1 / (1 + np.exp((self.mean - price) / self.scale))

    def push(self, price):
        if len(self.queue) < self.queue_length:
            self.queue.append(price)
            self.norm_queue.append(self.normalize(price))
        else:
            self.queue.append(price)
            self.queue = self.queue[1:]
            self.norm_queue.append(self.normalize(price))
            self.norm_queue = self.norm_queue[1:]

    def get_mean(self):
        return np.mean(np.array(self.queue))

    def get_norm_mean(self):
        return np.mean(np.array(self.norm_queue))

    def calc_next_day_norm_price(self, prediction):
        difference = self.queue_length * prediction - sum(self.norm_queue[1:])
        return difference

def initialize(context):
    pass


def load(universe, context):
    context.ii = 0
    stock_name = context.panel.axes[0][0]
    X = np.array(context.panel.loc[stock_name, :, 'price'])
    context.scale = np.var(X) ** 0.5
    context.mean = np.mean(X)

    context.N_HIDDEN = 100

    context.l_in = lasagne.layers.InputLayer(shape=(1, 1, 2))

    context.gate_parameters = lasagne.layers.recurrent.Gate(
        W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),
        b=lasagne.init.Constant(0.))

    context.cell_parameters = lasagne.layers.recurrent.Gate(
        W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),
        W_cell=None, b=lasagne.init.Constant(0.),
        nonlinearity=lasagne.nonlinearities.sigmoid)

    context.l_lstm = lasagne.layers.recurrent.LSTMLayer(
        context.l_in, context.N_HIDDEN,
        ingate=context.gate_parameters, forgetgate=context.gate_parameters,
        cell=context.cell_parameters, outgate=context.gate_parameters,
        learn_init=True, grad_clipping=100.)

    context.l_lstm_back = lasagne.layers.recurrent.LSTMLayer(
        context.l_in, context.N_HIDDEN,
        ingate=context.gate_parameters, forgetgate=context.gate_parameters,
        cell=context.cell_parameters, outgate=context.gate_parameters,
        learn_init=True, grad_clipping=100., backwards=True)

    context.l_sum = lasagne.layers.ElemwiseSumLayer([context.l_lstm,
                                                     context.l_lstm_back])

    context.n_batch, context.n_time_steps, context.n_features = context.l_in.input_var.shape
    context.l_reshape = lasagne.layers.ReshapeLayer(context.l_sum, (-1, context.N_HIDDEN))

    context.l_dense = lasagne.layers.DenseLayer(
        context.l_reshape, num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid)

    context.l_out = lasagne.layers.ReshapeLayer(context.l_dense, (context.n_batch, context.n_time_steps))

    context.target_values = T.vector('target_output')
    context.network_output = lasagne.layers.get_output(context.l_out)
    context.predicted_values = context.network_output[:, -1]
    context.cost = T.mean((context.predicted_values - context.target_values) ** 2)
    context.all_params = lasagne.layers.get_all_params(context.l_out)
    context.updates = lasagne.updates.adam(context.cost, context.all_params)

    context.train = theano.function([context.l_in.input_var,
                                     context.target_values],
                                    context.cost,
                                    updates=context.updates)

    context.compute_cost = theano.function(
        [context.l_in.input_var, context.target_values], context.cost)

    context.get_output = theano.function([context.l_in.input_var], context.network_output)

    with np.load(universe.load_file) as f:
         param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(context.l_out, param_values)

    context.history = My_history_with_weekends(5, context.mean, context.scale)
    for price in X[-5:]:
        context.history.push(price)
    #print(context.history.queue, context.history.norm_queue)

    context.trash_days_counter = 0
    context.current_day = len(X)

def handle_data(context, data):
    stock_name = context.panel.axes[0][0]

    def normalize(price):
        return 1 / (1 + np.exp((context.mean - price) / context.scale))

    def unnormalize(prediction):
        return context.mean - context.scale * (np.log((1 / prediction) - 1))

    context.current_day += 1
    context.trash_days_counter += 1
    current_price = data[symbol(stock_name)].price
    context.history.push(current_price)
    current_mean = context.history.get_mean()

    X = normalize(current_price)
    X_2 = context.history.get_norm_mean()
    input_ = np.array([[X, X_2]]).astype(theano.config.floatX).reshape(1, 1, 2)

    prediction = context.get_output(input_)[0][0]
    next_day_norm_price = context.history.calc_next_day_norm_price(prediction)
    unnorm_prediction = unnormalize(prediction)
    day2_input_ = np.array([[next_day_norm_price, prediction]])
    day2_input_ = day2_input_.astype(theano.config.floatX).reshape(1, 1, 2)
    prediction_2_day = context.get_output(day2_input_)[0][0]
    day2_norm_price = prediction_2_day * 5 - \
                      sum(context.history.norm_queue[2:]) - \
                      next_day_norm_price

    if context.current_day % 14 == 0:
        if max(context.history.norm_queue) < next_day_norm_price - 0.08:
            # print('buy',day2_norm_price, next_day_norm_price,
            #      max(context.history.norm_queue))
            order(symbol(stock_name), int(10000 * (next_day_norm_price - max(context.history.norm_queue))))
        elif next_day_norm_price + 0.08 < min(context.history.norm_queue):
            # print('sell', day2_norm_price, next_day_norm_price,
            #     min(context.history.norm_queue))
            order(symbol(stock_name), int(10000 * (next_day_norm_price - min(context.history.norm_queue))))

    record(normPrediction=prediction)
    record(normPrices=X_2)
    record(Prediction=unnorm_prediction)
    record(Prices=current_mean)


def analyze(universe=None, results=None):
    f, ax1 = plt.subplots(1, figsize=(12, 6))
    #ax1 = plt.subplot(311)

    #ax1 = plt.subplot(111)
    ax1.set_ylabel('Portfolio value (USD)')
    results.portfolio_value.plot(ax=ax1)
    plt.legend(loc=3)

    #ax2 = plt.subplot(312, sharex=ax1)
    #results.normPrices.plot(ax=ax2, label="normalized prices")
    #results.normPrediction.plot(ax=ax2, label="normalized net prediction")
    #plt.legend(loc=2)

    #ax3 = plt.subplot(313, sharex=ax2)
    #results.Prices.plot(ax=ax3, label="prices")
    #results.Prediction.plot(ax=ax3, label="net prediction")
    #plt.gcf().set_size_inches(27, 12)
    #plt.legend(loc=3)
    #plt.show()
    return f


def save(universe, context):
    pass
    #pd.to_pickle(context.result, universe.save_file)


