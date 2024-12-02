import logging
import numpy as np
from utils import *
from fsrank import *

def selection_phase(X, y, args):
    """
    Perform feature selection using multiple methods until stability criteria are met.

    Parameters:
    X (pd.DataFrame): Input features.
    y (pd.Series): Target variable.
    args (Namespace): Command-line arguments.

    Returns:
    str: Name of the most stable feature selection method.
    int: Number of features selected by the most stable method.
    """
    global logger
    has_found_stable_method = False
    best_stable_method = None
    best_metric_value = 0
    total_features = X.shape[1]
    increment = int(args.percent_increment * total_features)
    num_features = int(args.initial_features_percent * total_features)
    while num_features < (total_features + increment) and not has_found_stable_method:
        k = total_features if num_features > total_features else num_features
        print_message(f'Number of Features: {k}', 'info', logger)

        for method_id in methods.keys():
            print_message(f'Running {method_id}', 'info', logger)
            feature_scores = methods[method_id]['function'](X, y, k)
            new_X = X[list(feature_scores['features'])]
            metrics =  calculateMetrics(new_X, y)
            methods[method_id]['results'].append([k] + metrics)
            if len(methods[method_id]['results']) > 1:
                previous_metrics = methods[method_id]['results'][-2][1:]
                current_metrics = methods[method_id]['results'][-1][1:]
                if is_method_stable(logger, previous_metrics, current_metrics, args.threshold):
                    has_found_stable_method = True
                    #print(current_metrics)
                    accuracy = current_metrics[0]
                    if(accuracy > best_metric_value):
                        best_metric_value = accuracy
                        best_stable_method = method_id
        num_features += increment

    if(not has_found_stable_method):
        best_stable_method = choice(list(methods.keys()))

    k = int(methods[best_stable_method]['results'][-1][0])
    return best_stable_method, k

#def run(args, dataset):
def run(args, X, y):
    """
    Main function to execute feature selection process and save reduced dataset.

    Parameters:
    args (Namespace): Command-line arguments.
    X (pd.DataFrame): Input features.
    y (pd.Series): Target variable.

    Returns:
    bool: True if feature selection and dataset saving were successful.
    """
    global logger
    logging.basicConfig(format = '%(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('Original')
    logger.setLevel(logging.INFO)
    print_message('Starting Features Selection', 'info', logger)
    best_stable_method, lower_bound = selection_phase(X, y, args)
    return best_stable_method, lower_bound
