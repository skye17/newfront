import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Define algorithm
def load(universe, context):
    context.result = pd.read_pickle(universe.load_file)

def save(universe, context):
    pd.to_pickle(context.result, universe.save_file)

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