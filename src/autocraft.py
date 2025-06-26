import logging
import math
import numpy as np
from random import choice
from utils import *
from fsrank import *

def distribution(df, metric, step, max_value = 1.0):
    count = 0
    min_features = 0
    data_list = list()
    for i in np.arange(0.0, 1.01, step):
        th = max_value * i
        selected_ft = df[df[metric] <= th]
        ic = len(selected_ft) - count
        if not ic:
            continue
        fc = ic / len(df)
        afc = len(selected_ft) / len(df)
        count = len(selected_ft)
        pi = i * 100.0
        fc *= 100.0
        afc *= 100.0
        data_list.append([pi, fc, afc, ic])
        if pi >= 30.0:
            min_features = min_features + ic
        if afc == 100.0:
            break
    return min_features

def score_mig(features, target):
    features_id = features.columns
    mutualInformationGain = mutual_info_classif(features, target, random_state = 0)
    data = {'feature': features_id, 'score': mutualInformationGain}
    df = pd.DataFrame(data)
    df = df.sort_values(by = ['score'], ascending = False)
    return df

def score_pca(features):
    features_id = features.columns
    n_components = 3
    pca = PCA(n_components)
    pca.fit(features)
    score = pca.components_.T * np.sqrt(pca.explained_variance_)[np.newaxis, :]
    data = {'feature': features_id, 'score': np.abs(score).mean(axis = 1)}
    df = pd.DataFrame(data)
    df = df.sort_values(by = ['score'], ascending = False)
    return df

def n_by_score(X, y, function):
    global logger
    if function == 'mig':
        ft_score = score_mig(X, y)
    elif function == 'pca':
        ft_score = score_pca(X)
    max_score = ft_score['score'].max()
    print_message(f'Maximum {function.upper()} Score: {max_score:.5f}', 'info', logger)
    n = distribution(ft_score, 'score', 0.1, max_score)
    print_message(f'{function.upper()} Minimum Number of Features: {n}', 'info', logger)
    return n

def distance(coordinates):
    """
    Calculates the Euclidean distance between a set of coordinates and the origin (0, 0, ..., 0).

    Parameters:
        coordinates (list): List of coordinates (x1, x2, ..., xn).

    Returns:
        float: The Euclidean distance between the point and the origin.
    """
    sum_of_squares = sum(x**2 for x in coordinates)
    return math.sqrt(sum_of_squares)

def radar_chart_area(points):
    """
    Calculates the area of a radar chart for a given number of points (radii).

    Parameters:
        points (list): List of radii (distances from the center to the points).

    Returns:
        float: Area of the polygon.
    """
    n = len(points)
    if n < 3:
        raise ValueError('The function requires at least 3 radii to form a polygon')

    # Calculate equally spaced angles in radians
    angles = [2 * math.pi * i / n for i in range(n)]
    # Close the polygon by repeating the first radius
    points.append(points[0])
    angles.append(angles[0])
    # Calculate the area using the formula
    sum = 0.0
    for i in range(n):
        r_i = points[i]
        r_i1 = points[i + 1]
        theta_i = angles[i]
        theta_i1 = angles[i + 1]
        sum += r_i * r_i1 * math.sin(theta_i1 - theta_i)
    # Divide by 2 and return the absolute value
    area = abs(sum)/2
    return area

def median(values):
    values.sort()
    m = (values[1] + values[2])/2
    return m

def start_value(initial, increment, final):
    """
    Finds the largest number by repeatedly adding the increment to the initial value,
    but staying smaller than the final value.

    Args:
        initial (int): The initial number.
        increment (int): The increment value.
        final (int): The final threshold value.

    Returns:
        int: The largest number less than the final value.
    """
    actual = initial
    while actual + increment < final:
        actual += increment
    return actual

def selection_phase(X, y, n_min, args):
    """
    Perform feature selection using multiple methods until stability criteria are met.

    Parameters:
    X (pd.DataFrame): Input features.
    y (pd.Series): Target variable.
    n_min (int): Minimum number of features expected
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
    #num_features = int(args.initial_features_percent * total_features)
    num_features = start_value(0, increment, n_min)
    #num_features = args.initial_n_features
    #num_features = n_min - 1
    #increment = args.increment
    comparison_metrics = {
        'median': median,
        'area': radar_chart_area,
        'distance': distance
    }

    while num_features < (total_features + increment) and not has_found_stable_method:
        k = total_features if num_features > total_features else num_features
        print_message(f'Number of Features: {k}', 'info', logger)

        if args.parallelize:
            print_message('Running Methods in Parallel', 'info', logger)
            results = execute_methods_in_parallel(X, y, k, logger)
        for method_id in methods.keys():
            if args.parallelize:
                feature_scores = results[method_id]
            else:
                print_message(f'Running {method_id}', 'info', logger)
                feature_scores = methods[method_id]['function'](X, y, k)
            new_X = X[list(feature_scores['features'])]
            metrics = calculateMetrics(new_X, y)
            methods[method_id]['results'].append([k] + metrics)
            if len(methods[method_id]['results']) > 1:
                previous_metrics = methods[method_id]['results'][-2][1:]
                current_metrics = methods[method_id]['results'][-1][1:]
                #print('current', current_metrics)
                if num_features >= n_min and is_method_stable(logger, previous_metrics, current_metrics, args.threshold):
                    has_found_stable_method = True
                    # >>>>> <<<<< #
                    comparison_metric = comparison_metrics[args.metric](current_metrics)
                    #print('metric', comparison_metric)
                    if comparison_metric > best_metric_value:
                        best_metric_value = comparison_metric
                        best_stable_method = method_id
                    # >>>>> <<<<< #
        num_features += increment

    if not has_found_stable_method:
        best_stable_method = choice(list(methods.keys()))

    k = int(methods[best_stable_method]["results"][-1][0])
    return best_stable_method, k

def feature_analysis(X, y):
    global logger
    n = int(0.15 * X.shape[1])
    print_message(f'Minimum Number of Features Expected: {n}', 'info', logger)
    n_mig = n_by_score(X, y, 'mig')
    n_pca = n_by_score(X, y, 'pca')
    test_mig = n <= n_mig < 2 * n
    test_pca = n <= n_pca < 2 * n
    if test_mig or test_pca:
        n = n_mig if test_mig else n_pca
    elif test_mig and test_pca:
        n = n_mig if n_mig < n_pca else n_pca
    return n

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
    logger = logging.getLogger('Autocraft')
    logger.setLevel(logging.INFO)
    print_message('Starting Features Analysis', 'info', logger)
    n_min = feature_analysis(X, y)
    print_message('Starting Features Selection', 'info', logger)
    best_stable_method, lower_bound = selection_phase(X, y, n_min, args)
    return best_stable_method, lower_bound
