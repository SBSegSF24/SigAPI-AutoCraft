
#!/usr/bin/python3

import pandas as pd
import argparse
import sys

def parse_args(argv):
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-d', '--dataset', metavar='str',
        help='Dataset (csv file).', type=str, required=True)
    args = parser.parse_args(argv)
    return args

def calcula_grau_de_esparsividade(dataset):

        # gera coluna ZeroCount com contagem de 0s por linha
        dataset['ZeroCount'] = dataset.eq(0).sum(axis=1)

        # gera coluna ZeroCount com contagem de 1s por linha
        dataset['NonZeroCount'] = dataset.eq(1).sum(axis=1)

        # soma das somas de 0s da coluna ZeroCount
        n_zeros = dataset['ZeroCount'].sum()

        # soma das somas de 1s da coluna NonZeroCount
        n_nao_zeros = dataset['NonZeroCount'].sum()

        print("Dataset: " + args.dataset)

        print("Total de 0s: " + str(n_zeros))
        print("Total de 1s: " + str(n_nao_zeros))

        grau_de_esparsividade = (n_zeros/(n_zeros + n_nao_zeros))*100
        print('Grau de Esparsidade (GE): {:.2f}'.format(grau_de_esparsividade), "%")

if __name__=="__main__":
    try:
        args = parse_args(sys.argv[1:])

        #cria um dataframe a partir de um arquivo .csv
        dataset = pd.read_csv(args.dataset, sep=',')

        calcula_grau_de_esparsividade(dataset)
        
    except BaseException as e:
        print(e)
        exit(1)