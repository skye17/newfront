from zipline.api import order, record, symbol

# Define algorithm
def initialize(context):
    context.i = 0


def handle_data(context, data):
    for symb in map(symbol, context.panel.axes[0].values):
        if context.i == 0:
            order(symb, 10)
            context.i = 1

