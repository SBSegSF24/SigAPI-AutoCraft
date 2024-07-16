import sys
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from argparse import ArgumentParser
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Define the path to the directory
directory_path = Path('../tcc_laura_2022/PCA vs InfoGain')

def get_base_parser():
    parser = ArgumentParser(add_help=False)
    parser.add_argument('-d', '--dataset', type=str, required=True,
                        help='Dataset (csv file). It should be already preprocessed.')
    parser.add_argument('--sep', metavar='SEPARATOR', type=str, default=',',
                        help='Dataset feature separator. Default: ","')
    parser.add_argument('-c', '--class-column', type=str, default="class", metavar='CLASS_COLUMN',
                        help='Name of the class column. Default: "class"')
    parser.add_argument('-n', '--n-samples', type=int,
                        help='Use a subset of n samples from the dataset. By default, all samples are used.')
    parser.add_argument('-o', '--output-file', metavar='OUTPUT_FILE', type=str, default='results.csv',
                        help='Output file name. Default: results.csv')
    return parser

def parse_args(argv):
    base_parser = get_base_parser()
    parser = ArgumentParser(parents=[base_parser])
    parser.add_argument('-t', '--threshold', type=float, default=0.03,
                        help='Threshold for the difference between metrics at each increment on the number of features. When all metrics are less than it, the selection phase finishes. Default: 0.03')
    parser.add_argument('-f', '--initial-n-features', type=int, default=1,
                        help='Initial number of features. Default: 1')
    parser.add_argument('-i', '--increment', type=int, default=1,
                        help='Value to increment the initial number of features. Default: 1')
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
    X = dataset.drop(columns=parsed_args.class_column)
    y = dataset[parsed_args.class_column]
    return X, y

def calculatePCA(args, X):
    # Padronizando os dados (média=0 e variância=1)
    scaler = StandardScaler()
    dados_padronizados = scaler.fit_transform(X)

    # Realizando o PCA com 3 componentes 
    n_components = 3
    pca = PCA(n_components)
    componentes_principais = pca.fit_transform(dados_padronizados)

    # Atribuindo pontuações para cada característica com base na relevância (pesos)
    # Quanto maior o peso absoluto, mais relevante é a característica
    pontuacoes = pca.components_.T * np.sqrt(pca.explained_variance_)

    # Converter os arrays de pontuações em valores escalares (média dos elementos absolutos)
    feature_scores = {}
    for i, score in enumerate(pontuacoes):
        feature_name = X.columns[i]
        feature_scores[feature_name] = np.mean(np.abs(score))

    # Ordenar o dicionário com base nos valores dos scores (maior para menor)
    sorted_feature_scores = {k: v for k, v in sorted(feature_scores.items(), key=lambda item: item[1], reverse=True)}

    # Imprimir os scores ordenados
    print("Scores das características (maior para menor):")
    for i, (feature, score) in enumerate(sorted_feature_scores.items(), start=1):
        print(f"{i}. {feature}: {score}")

    # Ensure the directory exists
    directory_path.mkdir(parents=True, exist_ok=True)

    # Salvar os scores ordenados em um arquivo CSV
    output_file_path = directory_path / f'PCA_{Path(args.dataset).stem}.csv'
    df = pd.DataFrame(sorted_feature_scores.items(), columns=["Característica", "Score"])
    df.to_csv(output_file_path, index=False)

if __name__ == "__main__":
    parsed_args = parse_args(sys.argv[1:])
    X, y = get_X_y(parsed_args, get_dataset(parsed_args))
    total_features = get_dataset(parsed_args).shape[1] - 1
    print(f'Número de Características >> {X.shape[1]}')

    calculatePCA(parsed_args, X)
