import numpy as np
from sklearn.neighbors import NearestNeighbors
import sys
import argparse
from argparse import ArgumentParser
import pandas as pd
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

def parse_args(argv):
    base_parser = get_base_parser()
    parser = ArgumentParser(parents=[base_parser])
    parser.add_argument('-t', '--threshold', type = float, default = 0.03,
        help = 'Threshold for the difference between metrics at each increment on the number of features. When all metrics are less than it, the selection phase finishes. Default: 0.03')
    parser.add_argument( '-f', '--initial-n-features', type = int, default = 1,
        help = 'Initial number of features. Default: 1')
    parser.add_argument( '-i', '--increment', type = int, default = 1,
        help = 'Value to increment the initial number of features. Default: 1')
    args = parser.parse_args(argv)
    return args

def get_base_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument( '-d', '--dataset', type = str, required = True,
        help = 'Dataset (csv file). It should be already preprocessed.')
    parser.add_argument( '--sep', metavar = 'SEPARATOR', type = str, default = ',',
        help = 'Dataset feature separator. Default: ","')
    parser.add_argument('-c', '--class-column', type = str, default="class", metavar = 'CLASS_COLUMN', 
        help = 'Name of the class column. Default: "class"')
    parser.add_argument('-n', '--n-samples', type=int,
        help = 'Use a subset of n samples from the dataset. By default, all samples are used.')
    parser.add_argument('-o', '--output-file', metavar = 'OUTPUT_FILE', type = str, default = 'results.csv', 
        help = 'Output file name. Default: results.csv')
    return parser

def get_X_y(args, dataset):
    if(args.class_column not in dataset.columns):
        print(f'Class Column {args.class_column} Not Found in Dataset {args.dataset}')
        return None, None
    X = dataset.drop(columns = args.class_column)
    y = dataset[args.class_column]
    return X, y

def get_dataset(parsed_args):
    dataset = pd.read_csv(parsed_args.dataset, sep=parsed_args.sep)
    n_samples = parsed_args.n_samples
    if(n_samples):
        if(n_samples <= 0 or n_samples > dataset.shape[0]):
            raise Exception(f"Expected n_samples to be in range (0, {dataset.shape[0]}], but got {n_samples}")
        dataset = dataset.sample(n=n_samples, random_state=1, ignore_index=True)
    return dataset

def random_under_sampler(X, y):
    # Calcule o número de amostras de cada classe
    unique_classes, class_counts = np.unique(y, return_counts=True)

    # Encontre o rótulo da classe minoritária
    minority_class_label = unique_classes[np.argmin(class_counts)]
    minority_class_samples = np.min(class_counts)

    # Instancie o RandomUnderSampler para reduzir a classe majoritária
    undersample = RandomUnderSampler(sampling_strategy={label: minority_class_samples for label in unique_classes}, random_state=42)

    # Ajuste o RandomUnderSampler aos seus dados de treinamento
    X_resampled, y_resampled = undersample.fit_resample(X, y)

    # Crie um DataFrame temporário para armazenar a coluna de classes
    class_column = pd.DataFrame(y_resampled, columns=['class'], index=X_resampled.index)

    # Concatene o DataFrame temporário com o DataFrame original
    X_resampled = pd.concat([pd.DataFrame(X_resampled), class_column], axis=1)

    # Verifique os novos tamanhos das classes
    unique, counts = np.unique(y_resampled, return_counts=True)
    print(dict(zip(unique, counts)))
    return X_resampled, y_resampled
    
def random_under_sampler_reduced(X, y):
    # Calcule o número de amostras da classe minoritária
    minority_class_label = 1  # Substitua pelo rótulo real da classe minoritária
    minority_class_samples = np.sum(y == minority_class_label)

    # Determine o número de amostras que você deseja manter em cada classe
    target_samples = 5000

    # Instancie o RandomUnderSampler para reduzir ambas as classes para o número desejado de amostras
    undersample = RandomUnderSampler(sampling_strategy={0: target_samples, 1: target_samples}, random_state=42)

    # Ajuste o RandomUnderSampler aos seus dados de treinamento
    X_resampled, y_resampled = undersample.fit_resample(X, y)

    # Verifique os novos tamanhos das classes
    unique, counts = np.unique(y_resampled, return_counts=True)
    print(dict(zip(unique, counts)))
    return X_resampled, y_resampled

if __name__=="__main__":
    parsed_args = parse_args(sys.argv[1:])
    X, y = get_X_y(parsed_args, get_dataset(parsed_args))
    total_features = get_dataset(parsed_args).shape[1] - 1
    print("Nº total de features: ", total_features)

    dataset, target = random_under_sampler(X, y)
    dataset['class'] = target 
    dataset.to_csv(parsed_args.output_file, index=False)

    filename = parsed_args.output_file
    print(filename)
