from zipline.api import order, record, symbol
import matplotlib.pyplot as plt

# Define algorithm
def initialize(context):
    # Here we can use data from csv and write learned parameters into context's fields.
    context.panel  # Panel with training data.
    context.result = 12  # Here we can save result for serialization.
    context.previous = 0

def handle_data(context, data):
    context.panel  # Here we have access to training data also.
    stock_name = context.panel.axes[0][0]
    # Make solution using the result of learning:
    if not int(data[symbol(stock_name)].price) % context.result:
        order(symbol(stock_name), 10)
    # Record some values for analysis in 'analyze()'.
    sids = context.panel.axes[0].values
    prices = [data[symbol(sid)].price for sid in sids]
    record(Prices=prices)
    record(Prediction=3 * data[symbol(stock_name)].price - 2.2 * context.previous)
    # Record current price to use it in future.
    context.previous = data[symbol(stock_name)].price

## New default analyze function
def analyze(results, universe):
    f, (ax1, _) = plt.subplots(1, 2)
    ax1 = plt.subplot(211)
    results.portfolio_value.plot(ax=ax1)
    ax1.set_ylabel('Portfolio value (USD)')
    if len(universe.symbols) > 6:
        stocks = str(len(universe.symbols)) + " different stocks"
    else:
        stocks = ", ".join(universe.symbols)
    stocks_info = "Stocks: " + stocks
    f.text(.02, .02, ("Max portfolio value:{}\nMin portfolio value:{}\nEarned money:{}\n" + stocks_info).format(results.portfolio_value.max(),
                    results.portfolio_value.min(), results.portfolio_value.iloc[-1] - results.portfolio_value.iloc[0]))
    #plt.legend()
    return f