from zipline.api import order_percent, symbol
from sklearn.linear_model import LinearRegression
import numpy as np

# Define algorithm
def initialize(context):
    context.prev_prices = dict()


def handle_data(context, data):
    for symb in filter(data.__contains__, map(symbol, context.panel.axes[0].values)):
        prev = context.prev_prices.setdefault(symb, [])
        prev.append(data[symb].price)
        del prev[:-500]

        if len(prev) > 5:
            lr = LinearRegression().fit(np.arange(len(prev)).reshape((-1, 1)), prev)
            pred = lr.predict(np.asarray([len(prev)]).reshape((-1, 1)))[0]
            to_order = (prev[-1] - pred) / prev[-1] * 10.
            if abs(to_order) > .1:
                to_order /= 10. * abs(to_order)
            order_percent(symb, to_order)

