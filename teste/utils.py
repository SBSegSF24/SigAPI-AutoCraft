import os
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from termcolor import colored

def print_message(info, type, logger):
    """
    Prints a colored message to the console and logs it.

    Parameters:
    info (str): The message to print and log.
    type (str): The type of message ('warn', 'info', 'except', 'error').
    logger (Logger): The logger instance to use.
    """
    if type == 'warn':
        message = colored(f'{info}', 'yellow')
        logger.warning(message)
    elif type == 'info':
        message = colored(f'{info}', 'green')
        logger.info(message)
    elif type == 'except':
        message = colored(f'{str(info)}', 'red')
        logger.exception(message)
    elif type == 'error':
        message = colored(f'{str(info)}', 'red')
        logger.error(message)
    else:
        logger.info(info)

def get_dataset(dataset, logger):
    """
    Loads a dataset from a CSV file.

    Parameters:
    dataset (str): Path to the dataset file.
    logger (Logger): The logger instance to use.

    Returns:
    DataFrame: Loaded dataset as a pandas DataFrame, or None if loading fails.
    """
    if not os.path.isfile(dataset):
        print_message(f'Dataset File {dataset} Not Found', 'error', logger)
        return None
    try:
        dataset_df = pd.read_csv(dataset, low_memory = False)
    except Exception as e:
        print_message(e, 'except', logger)
        return None
    return dataset_df

def get_X_y(args, dataset, logger):
    """
    Splits the dataset into features (X) and labels (y).

    Parameters:
    args (Namespace): Command-line arguments.
    dataset (DataFrame): The loaded dataset.
    logger (Logger): The logger instance to use.

    Returns:
    Tuple: A tuple containing the feature set (X) and the label set (y).
    """
    if args.class_column not in dataset.columns:
        message = f'Dataset Does Not Have a Column Called "{args.class_column}"'
        print_message(message, 'error', logger)
        exit(1)
    X = dataset.drop(columns = args.class_column)
    y = dataset[args.class_column]
    return X, y

def check_directory(dir):
    """
    Checks if a directory exists and creates it if it does not.

    Parameters:
    dir (str): The directory path to check and create if necessary.
    """
    root_path = os.getcwd()
    dir_path = os.path.join(root_path, dir)
    if not os.path.exists(dir_path):
        os.makedirs(dir)

def calculateMetrics(new_X, y):
    new_X_train, new_X_test, y_train, y_test = train_test_split(new_X, y, test_size = 0.3, random_state = 0)
    clf = RandomForestClassifier(random_state = 0)
    clf.fit(new_X_train, y_train)
    prediction = clf.predict(new_X_test)
    accuracy = accuracy_score(y_test, prediction)
    precision = precision_score(y_test, prediction, zero_division = 0)
    recall = recall_score(y_test, prediction, zero_division = 0)
    f1 = f1_score(y_test, prediction, zero_division = 0)
    metrics = [accuracy, precision, recall, f1]
    return metrics

def is_method_stable(logger, previous_metrics, current_metrics, t = 0.03):
    """
    Check if a feature selection method is stable based on metric differences.

    Parameters:
    previous_metrics (list): Metrics of the previous iteration.
    current_metrics (list): Metrics of the current iteration.
    t (float): Threshold for metric differences.

    Returns:
    bool: True if method is stable, False otherwise.
    """
    differences = abs(np.array(current_metrics) - np.array(previous_metrics))
    print_message(f'Differences: {differences}', 'info', logger)
    if(all(differences < t)):
        return True
    return False
