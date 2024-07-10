import pandas as pd
import argparse
import sys

def get_X_y(args, dataset):
    if(args.class_column not in dataset.columns):
        print(f'Class Column {args.class_column} Not Found in Dataset {args.dataset}')
        return None, None
    X = dataset.drop(columns = args.class_column)
    y = dataset[args.class_column]
    return X, y

def parse_args(argv):
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-d', '--dataset', metavar='str',
        help='Dataset (csv file).', type=str, required=True)
    parser.add_argument(
        '--class-column', type = str,
        default = 'class', metavar = 'CLASS_COLUMN',
        help = 'Class Column ID. Default: class')
    args = parser.parse_args(argv)
    return args

def frequecia_caracteristicas(X_dataset):
    qtd_linhas = len(X_dataset.index)

    lista_colunas = X_dataset.columns
    list_freq = []
    for coluna in lista_colunas:
        #conta quantas vezes cada valor aparece por coluna
        a = X_dataset[coluna].value_counts() 
        #preenche a lista de frequencia
        list_freq.append([coluna, a[1]])

    #cria novo dataframe com a soma de vezes que uma característica aparece    
    df = pd.DataFrame(list_freq, columns = ['Caracteristica', 'Frequencia'])

    #cria arquivo .csv 
    df.to_csv(f'Frequencia_Caracteristicas_{args.dataset}')

    df = df.sort_values(by='Frequencia', ascending = False)
    df.index = range(len(df.index))
    
    for i in range(len(df.index)):
        frequencia =  (df['Frequencia'][i]/qtd_linhas)*100
        print(df['Caracteristica'][i], '{:.2f}'.format(frequencia), '%')
    

if __name__=="__main__":
    try:
        args = parse_args(sys.argv[1:])

        #cria um dataframe a partir de um arquivo .csv
        dataset = pd.read_csv(args.dataset, sep=',')

        X_dataset, y_dataset = get_X_y(args, dataset)

        frequecia_caracteristicas(X_dataset)
        
    except BaseException as e:
        print(e)
        exit(1)