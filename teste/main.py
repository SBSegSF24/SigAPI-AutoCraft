import time
import os
import pandas as pd
import numpy as np
import sys
import argparse
#from sklearn.model_selection import train_test_split
#from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
#from sklearn.decomposition import PCA
#from sklearn.feature_selection import SelectFromModel, mutual_info_classif, RFE, SelectKBest, chi2
#from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#from random import choice
from argparse import ArgumentParser
from utils import *
import matplotlib.pyplot as plt
from sklearn import metrics
from fsrank import methods
from importlib import import_module
#from original import *
#from autocraft import *
import logging

def float_range(mini,maxi):
    """
    Creates a function to check if a float argument is within a specific range.

    Parameters:
    mini (float): Minimum value of the range (exclusive).
    maxi (float): Maximum value of the range (exclusive).

    Returns:
    function: A function that checks if a float is within the given range.
    """
    def float_range_checker(arg):
        try:
            f = float(arg)
        except ValueError:
            raise argparse.ArgumentTypeError("Must be a Floating Point Number")
        if f <= mini or f >= maxi:
            raise argparse.ArgumentTypeError("Must be > " + str(mini) + " and < " + str(maxi))
        return f
    return float_range_checker

class DefaultHelpParser(argparse.ArgumentParser):
    """
    Custom argument parser that prints help message on error and logs the error.
    """
    def error(self, message):
        global logger
        self.print_help()
        msg = colored(message, 'red')
        logger.error(msg)
        sys.exit(2)

def parse_args(argv):
    """
    Parses command line arguments and returns them.

    Parameters:
    argv (list): Command line arguments.

    Returns:
    argparse.Namespace: Parsed arguments.
    """

    parser = DefaultHelpParser(formatter_class = argparse.RawTextHelpFormatter)
    parser._optionals.title = 'Optional Arguments'

    parser.add_argument(
        '-d', '--dataset', metavar = 'DATASET', help = 'Dataset (csv Files)',
        type = str,  required = True)
    parser.add_argument(
        '-c', '--class-column', type = str, default = 'class', metavar = 'CLASS_COLUMN',
        help = 'Name of Class Column. Default: class')
    parser.add_argument(
        '--parallelize', help = 'Parallel Execution',
        action = 'store_true')
    parser.add_argument(
        '--output', help = 'Output File Directory. Default: results',
        type = str, default = 'results')
    parser.add_argument(
        '-th', '--threshold', type = float_range(0.0, 1.0), default = 0.03,
        help = 'Threshold of Difference Between Metrics at Each Increment in Number of Features. When All Metrics Are Less Than It, Selection Phase Finishes. Default: 0.03')
    #parser.add_argument( '-f', '--initial-n-features', type = int, default = 1,
    #    help = 'Initial number of features. Default: 1')
    #parser.add_argument( '-i', '--increment', type = int, default = 1,
    #    help = 'Value to increment the initial number of features. Default: 1')

    parser.add_argument(
        '-ifp', '--initial-features-percent', type = float_range(0.0, 1.0), default = 0.05,
        help = 'Initial Features Percentage. Default: 0.02')
    parser.add_argument(
        '-pi', '--percent-increment', type = float, default = 0.05,
        help = 'Percentage to Increment Number of Features. Default: 0.02')

    parser.add_argument(
        '--autocraft', help = f'Run SigAPI Autocraft', action = 'store_true')
    parser.add_argument(
        '-m', '--metric', metavar = 'METRIC', default = 'median',
        help = 'Metric to Compare. Default: median. Choices: ' + str(['median', 'area', 'distance']),
        choices = ['median', 'area', 'distance'], type = str)
    args = parser.parse_args(argv)
    return args

def correlation_phase(X, y, k, method, methods, args):
    """
    Perform correlation analysis and drop highly correlated features.

    Parameters:
    X (pd.DataFrame): Input features.
    y (pd.Series): Target variable.
    k (int): Number of features to keep.
    method (str): Name of the feature selection method.
    methods (dict): Dictionary of feature selection methods and their details.
    args (Namespace): Command-line arguments.

    Returns:
    pd.DataFrame: Reduced dataset after dropping correlated features.
    """
    global logger
    feature_scores = methods[method]['function'](X, y, k)
    new_X = X[list(feature_scores['features'])]
    correlation = new_X.corr()
    model_RF = RandomForestClassifier()
    model_RF.fit(new_X,y)
    feats = dict()
    for feature, importance in zip(new_X.columns, model_RF.feature_importances_):
        feats[feature] = importance
    to_drop = set()

    for index in correlation.index:
        for column in correlation.columns:
            if index != column and correlation.loc[index, column] > 0.85:
               ft = column if feats[column] <= feats[index] else index
               to_drop.add(ft)

    print_message(f'Number of Features Removed: {len(to_drop)}', 'info', logger)
    reduced_dataset = new_X.drop(columns = to_drop)
    reduced_dataset[args.class_column] = y
    return reduced_dataset

def get_moving_average(data, window_size = 5):
    """
    Compute the moving average of data.

    Parameters:
    data (np.array): Array of data.
    window_size (int): Size of the moving average window.

    Returns:
    np.array: Moving averages of the data.
    """
    cumsum_vec = np.cumsum(np.insert(data, 0, 0))
    return (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size

def get_minimal_range_suggestion(df, t = 0.001, window_size = 5):
    """
    Get a minimal range suggestion based on gradients of moving averages.

    Parameters:
    df (pd.DataFrame): Dataframe of values.
    t (float): Threshold for gradient differences.
    window_size (int): Size of the moving average window.

    Returns:
    int: Minimal range suggestion index.
    """
    moving_averages = np.array([get_moving_average(np.array(df)[:, i], window_size) for i in range(df.shape[1])]).T
    gradients = np.gradient(moving_averages, axis = 0)
    diffs = gradients[1:] - gradients[:-1]

    for i in range(len(diffs) - 1, 1, -1):
        if(any([diff > t for diff in diffs[i]])):
            return int(df.index[i])
    return -1

if __name__=="__main__":
    global logger
    start = time.time()
    logging.basicConfig(format = '%(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('SigAPI')
    logger.setLevel(logging.INFO)
    args = parse_args(sys.argv[1:])

    info = f"Loading Dataset From {colored(os.path.basename(args.dataset), 'blue')}"
    print_message(info, 'info', logger)
    dataset = get_dataset(args.dataset, logger)
    if dataset is None:
        print_message('Error Loading Dataset','errot', logger)
    try:
        check_directory(args.output)
        number_features = dataset.shape[1] - 1
        print_message(f'Number of Features: {number_features}', 'info', logger)

        used_sigapi = 'autocraft' if args.autocraft else 'original'
        sigapi = import_module(used_sigapi)

        X, y = get_X_y(args, dataset, logger)
        best_stable_method, lower_bound = sigapi.run(args, X, y)
        print_message(f'Smallest Lower Limit Found: {best_stable_method}, {lower_bound}', 'info', logger)
        print_message('Starting Correlation', 'info', logger)
        reduced_dataset = correlation_phase(X, y, lower_bound, best_stable_method, methods, args)
        output_file = os.path.join(args.output, f'{used_sigapi}_{os.path.basename(args.dataset)}')
        print_message('Saving Reduced Dataset', 'info', logger)
        reduced_dataset.to_csv(output_file, index = False)
        print_message('Finished SigAPI Features Selection', 'info', logger)
    except Exception as e:
        msg = f'Error in Execution: {e}'
        logger.exception(msg)

    stop = time.time()
    total_time = stop - start
    print(f'Time: {total_time: .2f} segundos.')
