import argparse
import pandas as pd

def get_base_parser():
    """
    Configura um parser de argumentos para uma aplicação de linha de comando que manipula um conjunto de dados.
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

    Argumentos:
    -d, --dataset : Caminho para o arquivo CSV contendo o conjunto de dados. O arquivo deve estar previamente processado.
    --sep  Separador de campo das características no conjunto de dados. O padrão é uma vírgula `,`.
    -c, --class-column : Nome da coluna que contém as classes ou rótulos no conjunto de dados. O padrão é "class".
    -n, --n-samples : Número de amostras a serem utilizadas do conjunto de dados. Por padrão, todas as amostras são utilizadas.
    -o, --output-file : Nome do arquivo de saída onde os resultados serão salvos. O padrão é `results.csv`.

    Retorna:
    -parser : ArgumentParser com os parâmetros comuns entre os métodos implementados. 
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
    """
    Carrega um conjunto de dados a partir de um arquivo CSV e realiza uma amostragem aleatória, se especificado.

    Parâmetros:
    - parsed_args: objeto do tipo Namespace contendo os seguintes atributos:
        - dataset: caminho para o arquivo CSV contendo o conjunto de dados.
        - sep: delimitador de campo utilizado no arquivo CSV (ex: ',' para vírgula).
        - n_samples: número de amostras aleatórias a serem selecionadas. Se None, todo o conjunto de dados é carregado.

    Retorna:
    - dataset (pandas.DataFrame): DataFrame contendo o conjunto de dados carregado.

    Exceções:
    - Exception: levantada se o número de amostras for menor ou igual a 0, ou se for maior do que o número total de linhas no conjunto de dados.
    """
    dataset = pd.read_csv(parsed_args.dataset, sep=parsed_args.sep)
    n_samples = parsed_args.n_samples
    if(n_samples):
        if(n_samples <= 0 or n_samples > dataset.shape[0]):
            raise Exception(f"Expected n_samples to be in range (0, {dataset.shape[0]}], but got {n_samples}")
        dataset = dataset.sample(n=n_samples, random_state=1, ignore_index=True)
    return dataset

def get_X_y(parsed_args, dataset):
    """
    Extrai as características (X) e os rótulos de classe (y) de um conjunto de dados fornecido.

    Parâmetros:
    - parsed_args: objeto do tipo Namespace contendo os seguintes atributos:
        - class_column (str): Nome da coluna que contém os rótulos de classe no conjunto de dados.
        - dataset (str): Caminho para o arquivo CSV que contém o conjunto de dados, utilizado para exibir na mensagem de erro, caso necessário.
    - dataset (pandas.DataFrame): DataFrame que contém o conjunto de dados a partir do qual as características (X) e os rótulos de classe (y) serão extraídos.

    Retorna:
    - X (pandas.DataFrame): DataFrame contendo todas as colunas do conjunto de dados original, exceto a coluna de classe especificada.
    - y (pandas.Series): Série contendo os rótulos de classe extraídos da coluna especificada no conjunto de dados.

    Exceções:
    - Exception: Levantada se a coluna de classe especificada em `parsed_args.class_column` não for encontrada no conjunto de dados.

    """
    if(parsed_args.class_column not in dataset.columns):
        raise Exception(f'Expected dataset {parsed_args.dataset} to have a class column named "{parsed_args.class_column}"')
    X = dataset.drop(columns = parsed_args.class_column)
    y = dataset[parsed_args.class_column]
    return X, y
