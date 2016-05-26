import matplotlib.pyplot as plt
from zipline.api import order, record, symbol
from scipy.optimize import leastsq
import numpy as np

class My_history_with_weekends(object):

    def __init__(self, queue):
        self.queue = queue
        self.middle = int(len(queue)/2)
        print(self.middle)
        assert len(self.queue) % 2 == 1

    def push(self, price):
        self.queue.append(price)
        self.queue = self.queue[1:]

    def contain_local_min(self):
        return self.queue[self.middle] == min(self.queue)

    def contain_local_max(self):
        return self.queue[self.middle] == max(self.queue)

    def get_middle(self):
        return self.queue[self.middle]


def find_local_minimums_and_maximums(data, number=10, neighb_radius=2):
    minimums = []
    maximums = []

    for i in range(len(data) - neighb_radius - 1, neighb_radius, - 1):
        if len(minimums) < number and data[i] == min(data[i - neighb_radius: i + neighb_radius]):
            minimums.append(i)

        elif len(maximums) < number and data[i] == max(data[i - neighb_radius: i + neighb_radius]):
            maximums.append(i)

        elif len(maximums) == number and len(minimums) == number:
            break

    minimums.sort()
    maximums.sort()

    return minimums, maximums


def line_appoximation(x, plsq):
    return plsq[0] + plsq[1]*x


def get_line(points, ts_values):

    def func(x, p):
        a, b = p
        return [a + b * t for t in x]

    def calc_error(p, y, x, func):
        return [(u - v) ** 2 for (u, v) in zip(y, func(x, p))]

    x_0 = [1., 1.]
    plsq = leastsq(calc_error, x_0, args=(ts_values, points, func))[0]
    return lambda x: line_appoximation(x, plsq)


def initialize(context):
    context.alpha = 0.7
    stock_name = context.panel.axes[0][0]
    data = np.array(context.panel.loc[stock_name, :, 'price'])
    context.data = np.array([data[0]] + list(context.alpha * data[1:] +
                                             (1 - context.alpha) * data[:-1]))

    context.mins, context.maxs = find_local_minimums_and_maximums(context.data)
    context.ts_mins = [context.data[m] for m in context.mins]
    context.ts_maxs = [context.data[m] for m in context.maxs]

    context.resistance_line = get_line(context.maxs, context.ts_maxs)
    context.support_line = get_line(context.mins, context.ts_mins)

    prehistory = list(context.panel.loc[stock_name, :, 'price'][-5:])
    context.history = My_history_with_weekends(prehistory)
    context.previous = prehistory[-1]
    context.current_day = len(context.data)
    context.use_support_line = True
    context.use_resistance_line = True


def handle_data(context, data):
    stock_name = context.panel.axes[0][0]
    context.current_day = context.current_day + 1
    current_price = data[symbol(stock_name)].price
    current_price_smoothed = context.alpha * current_price + (1 - context.alpha) * context.previous

    context.history.push(current_price_smoothed)

    if context.history.contain_local_min():
        context.mins.append(context.current_day - 2)
        context.mins = context.mins[1:]
        context.ts_mins.append(context.history.get_middle())
        context.ts_mins = context.ts_mins[1:]
        context.support_line = get_line(context.mins, context.ts_mins)

    elif context.history.contain_local_max():
        context.maxs.append(context.current_day - 2)
        context.maxs = context.maxs[1:]
        context.ts_maxs.append(context.history.get_middle())
        context.ts_maxs = context.ts_maxs[1:]
        context.resistance_line = get_line(context.maxs, context.ts_maxs)

    support_price = context.support_line(context.current_day)
    resistance_price = context.resistance_line(context.current_day)
    # print(support_price, resistance_price, current_price, context.current_day)

    if current_price > context.previous and resistance_price - current_price < 0.03:
        order(symbol(stock_name), -10)
    elif current_price < context.previous and current_price - support_price < 0.03:
        order(symbol(stock_name), 10)

    # elif current_price > resistance_price:
    #    order(symbol('AAPL'), 10)
    #    context.support_line = context.resistance_line
    # elif current_price < support_price:
    #    order(symbol('AAPL'), -10)
    #    context.resistance_line = context.support_line

    context.previous = current_price
    record(Prices=current_price)
    record(supportPrices=support_price)
    record(resistPrices=resistance_price)

def analyze(universe=None, results=None):
    # Here we can use recorded values and analyze object returned by run().
    # Standard part
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1 = plt.subplot(211)
    results.portfolio_value.plot(ax=ax1)
    ax1.set_ylabel('Portfolio value (USD)')

    plt.gcf().set_size_inches(18, 8)
    plt.legend()

    ax2 = plt.subplot(212, sharex=ax1)
    results.Prices.plot(ax=ax2, label="prices")
    results.supportPrices.plot(ax=ax2, label="support_prices")
    results.resistPrices.plot(ax=ax2, label="resist_prices")
    plt.legend(loc=2)
    return f
