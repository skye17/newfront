from zipline.api import order, record, symbol
import matplotlib.pyplot as plt

# Define algorithm
def initialize(context):
    context.i = 0


def handle_data(context, data):
    context.i += 1
    stock_name = context.panel.axes[0][0]
    if context.i == 60:
        order(symbol(stock_name), 10)
    if context.i == 150:
        order(symbol(stock_name), -10)
    record(Prices=data[symbol(stock_name)].price)


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
    plt.legend(loc=2)
    return f

