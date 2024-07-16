import sys
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from argparse import ArgumentParser

def get_base_parser():
    parser = ArgumentParser(add_help=False)
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

def get_dataset(parsed_args):
    dataset = pd.read_csv(parsed_args.dataset, sep=parsed_args.sep)
    n_samples = parsed_args.n_samples
    if(n_samples):
        if(n_samples <= 0 or n_samples > dataset.shape[0]):
            raise Exception(f"Expected n_samples to be in range (0, {dataset.shape[0]}], but got {n_samples}")
        dataset = dataset.sample(n=n_samples, random_state=1, ignore_index=True)
    return dataset

def get_X_y(parsed_args, dataset):
    if(parsed_args.class_column not in dataset.columns):
        raise Exception(f'Expected dataset {parsed_args.dataset} to have a class column named "{parsed_args.class_column}"')
    X = dataset.drop(columns = parsed_args.class_column)
    y = dataset[parsed_args.class_column]
    return X, y


def calculateRFE(X, y):
    # Criando o modelo de classificação (Regressão Logística)
    model = LogisticRegression()

    # Criando o objeto RFE (Recursive Feature Elimination) com o modelo e o número desejado de características a serem selecionadas
    num_features_to_select = 2  # Número de características a serem selecionadas
    rfe = RFE(estimator=model, n_features_to_select=num_features_to_select)

    # Ajustando o RFE ao conjunto de dados
    fit = rfe.fit(X, y)

    print("Número de características selecionadas:", fit.n_features_)
    print("Índices das características selecionadas:", fit.support_)
    print("Ranking de características (1 = selecionada, 2 = segunda melhor, etc.):", fit.ranking_)

if __name__=="__main__":
    parsed_args = parse_args(sys.argv[1:])
    X, y = get_X_y(parsed_args, get_dataset(parsed_args))
    total_features = get_dataset(parsed_args).shape[1] - 1
    print(f'Número de Características >> {X.shape[1]}')

    calculateRFE(X, y)
