import argparse
import pandas as pd

def get_base_parser():
    """
    Retorna um ArgumentParser com os parâmetros comuns entre os métodos implementados. 
    Para incluir os parâmetros deste parser em outro, passe-o para o outro parser da seguinte forma:
    ```
    from argparse import ArgumentParser
    from utils import get_base_parser

    base_parser = get_base_parser()
    other_parser = ArgumentParser(parents=[base_parser])

    # Adicione os parâmetros específicos do outro parser normalmente:
    other_parser.add_argument("-f", help="Lista de features", ...)
    other_parser.add_argument("-k", help="Qtd de folds na validação cruzada", ...)
    other_parser.add_argument(...)
    ```
    """
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