import pytz
import imp
import pandas as pd
from datetime import datetime
from zipline.algorithm import TradingAlgorithm
import default
import os, sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

def check_important_columns(data):
    for df in data:
        if 'volume' not in df.axes[1].values:
            default = 10000
            df['volume'] = default
            print "Can't find column 'volume' needed for calculations, " \
                  "column 'volume' with default values(%i) added" % default

        if 'price' not in df.axes[1].values:
            df['price'] = df['close']
            print "Can't find column 'price', column 'close' will be used as price"

def make_panels(universe):
    if universe.parser:
        def parser_func(x):
            return datetime.strptime(x, universe.parser)
        data = [pd.read_csv(filename, index_col=0, date_parser=parser_func) for filename in universe.data_files]
    else:
        data = [pd.read_csv(filename, index_col=0, parse_dates=True) for filename in universe.data_files]

    for dt in data:
        dt.index = dt.index.tz_localize(pytz.UTC)

    check_important_columns(data)

    if len(universe.symbols) == 0:
        universe.symbols = ["STOCK_" + str(i) for i in range(len(data))]

    dict_data = dict()
    for stock, dt in zip(universe.symbols, data):
        dict_data[stock] = dt

    pn = pd.Panel(dict_data)
    if universe.training_size < 1:
        training_size = int(pn.shape[1] * universe.training_size)
    else:
        training_size = int(universe.training_size)
    training_pn = pn[:, :training_size, :]
    test_pn = pn[:, training_size:, :]
    return training_pn, test_pn

def wrapped_init(context):
    context.panel = training_panel
    if universe.load_file != "":
        load_1(universe, context)
    else:
        initialize_1(context)
    if universe.save_file != "":
        save_1(universe,context)

class Universe:
    def __init__(self):
        self.algo_filename = ""
        self.data_files = ""
        self.save_file = ""
        self.load_file = ""
        self.frequency = ""
        self.capital_base = 10000
        self.parser = None
        self.symbols = []
        self.training_size = None

    def fill_parameters(self, params):
        paths = []
        for data_filename in params['data_filenames']:
            paths.append(os.path.join(BASE_DIR, 'uploads/datasets/'+data_filename))
        if params['load_file'] != "":
            self.load_file = os.path.join(BASE_DIR, 'uploads/'+params['load_file'])
        if params['save_file'] != "":
            self.save_file = os.path.join(BASE_DIR, 'uploads/'+params['save_file'])
        self.data_files = paths
        self.algo_filename = params['algorithm']
        self.frequency = params['frequency']
        self.capital_base = params['base_capital']
        self.parser = params['parser']
        self.training_size = params['train_size']



def handle_algorithm_testing(params):
    global universe, training_panel, test_panel
    universe = Universe()
    universe.fill_parameters(params)
    global load_1, save_1, initialize_1
    algo_filename = universe.algo_filename

    path = os.path.join(BASE_DIR, 'uploads/algorithms/'+algo_filename)
    pathUtils = os.path.join(BASE_DIR, 'uploads/algorithms/utils.py')
    pathDBN = os.path.join(BASE_DIR, 'uploads/algorithms/DBN.py')
    pathRBM = os.path.join(BASE_DIR, 'uploads/algorithms/RBM.py')
    pathCRBM = os.path.join(BASE_DIR, 'uploads/algorithms/CRBM.py')
    pathLog = os.path.join(BASE_DIR, 'uploads/algorithms/LogisticRegression.py')
    pathMLP = os.path.join(BASE_DIR, 'uploads/algorithms/MLP.py')

    sys.path.append(path)
    global user_defined_mod
    utils = imp.load_source("utils", pathUtils)
    LogisticRegression = imp.load_source("LogisticRegression", pathLog)
    MLP = imp.load_source("MLP", pathMLP)
    RBM = imp.load_source("RBM", pathRBM)
    CRBM = imp.load_source("CRBM", pathCRBM)
    DBN = imp.load_source("DBN", pathDBN)

    user_defined_mod = imp.load_source("user_defined_mod", path)

    training_panel, test_panel = make_panels(universe)

    handle_data_1 = user_defined_mod.handle_data

    initialize_1 = user_defined_mod.initialize

    if hasattr(user_defined_mod, 'analyze') and callable(getattr(user_defined_mod, 'analyze')):
        analyze_1 = user_defined_mod.analyze
    else:
        analyze_1 = default.analyze

    if hasattr(user_defined_mod, 'load') and callable(getattr(user_defined_mod, 'load')):
        load_1 = user_defined_mod.load
    else:
        load_1 = default.load

    if hasattr(user_defined_mod, 'save') and callable(getattr(user_defined_mod, 'save')):
        save_1 = user_defined_mod.save
    else:
        save_1 = default.save


    algo = TradingAlgorithm(initialize=lambda x: wrapped_init(x),
                            handle_data=handle_data_1,
                            data_frequency=universe.frequency,
                            capital_base=universe.capital_base,
                            )
    performance = algo.run(test_panel)


    result, portfolio = analyze_1(results=performance, universe = universe), performance.portfolio_value
    '''
    universe = None
    training_panel = None
    test_panel = None
    load_1 = None
    save_1 = None
    initialize_1 = None
    '''
    analyze_1 = None

    del sys.modules['user_defined_mod']

    return result,portfolio