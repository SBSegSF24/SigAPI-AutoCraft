import sys
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import logging
from termcolor import colored, cprint
#from ydata_quality import DataQuality

def parse_args(argv):
    """
    Analisa os argumentos da linha de comando para a execução do script.

    Parâmetros:
    - argv : Lista de strings representando os argumentos da linha de comando. Normalmente, isso é passado automaticamente pelo sistema quando o script é executado.

    Retorna:
    - args: Um objeto Namespace contendo os argumentos analisados e seus valores.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-d', '--dataset', metavar = 'DATASET',
        help = 'Dataset (csv file).', type = str, required = True)
    parser.add_argument(
        '--class-column', metavar = 'CLASS_COLUMN',
        help = 'Classification Column.', type = str, default = 'class')
    parser.add_argument(
        '--drop-class', help = 'Drop Class Column',
        action = 'store_true')
    parser.add_argument(
        '--verbose', help = 'Increase Output Data.',
        action = 'store_true')
    args = parser.parse_args(argv)
    return args

def print_info(message):
    print(colored(message, 'green'))

def get_X_y(args, dataset):
    """
    Extrai as características (X) e o rótulo de classe (y) do conjunto de dados com base na coluna de classe fornecida.

    Parâmetros:
    - args : Objeto contendo os argumentos analisados, incluindo o nome da coluna de classe.
    - dataset : Conjunto de dados carregado em um DataFrame do pandas.

    Retorna:
    - X : DataFrame contendo todas as colunas do conjunto de dados, exceto a coluna de classe.
    - y :Série contendo os valores da coluna de classe do conjunto de dados.
    """
    if(args.class_column not in dataset.columns):
        print(f'Class Column {args.class_column} Not Found in Dataset {args.dataset}')
        return None, None
    X = dataset.drop(columns = args.class_column)
    y = dataset[args.class_column]
    return X, y

def calculate_mig_features(features, target):
    """
    Calcula o ganho de informação mútua (Mutual Information Gain) para as características fornecidas e retorna um DataFrame contendo a importância de cada característica.

    Parâmetros:
    - features : DataFrame contendo as características (features) do conjunto de dados, onde cada coluna representa uma característica.
    - target : Série contendo os rótulos de classe ou variáveis alvo associadas às amostras em `features`.

    Retorna:
    - df: DataFrame contendo a importância das características calculada pelo ganho de informação mútua. O DataFrame possui duas colunas:
        - "feature": Nome das características.
        - "score": Valor do ganho de informação mútua associado a cada característica.
    """   
    features_names = features.columns
    mutualInformationGain = mutual_info_classif(features, target, random_state = 0)
    data = {'feature': features_names, 'score': mutualInformationGain}
    df = pd.DataFrame(data)
    df = df.sort_values(by=['score'], ascending=False)
    return df

def mig_features(dataset, args):
    """
    Avalia  o ganho de Informação Mútua (Mutual Information Gain) das características de um conjunto de dados e exibe a distribuição dessas pontuações.

    Parâmetros:
    - dataset : DataFrame contendo o conjunto de dados que inclui as características e a coluna de classe.
    - args : Objeto contendo os argumentos de entrada que definem parâmetros como a coluna de classe e outros parâmetros necessários para a função `get_X_y`.
    """
    print_info('>>> Evaluating Information Gain <<<')
    X_dataset, y_dataset = get_X_y(args, dataset)
    mig = calculate_mig_features(X_dataset, y_dataset)
    max_ig = mig['score'].max()
    print(f'Max. IG {max_ig:.5f}')
    print_distribution(mig, 'score', 0.1, max_ig)

def duplicate_samples(dataset, args):
    """
    Avalia a quantidade de amostras duplicadas em um conjunto de dados e calcula a porcentagem de redução ao remover duplicatas.

    Parâmetros:
    - dataset : DataFrame contendo o conjunto de dados a ser analisado, incluindo as características.
    - args : Objeto contendo os argumentos de entrada que definem parâmetros como a coluna de classe e se a coluna de classe deve ser descartada.
    """
    print_info('>>> Evaluating Duplicate Samples <<<')
    ds = dataset
    if args.drop_class:
        ds = dataset.drop(columns=[args.class_column])
    print(f'Number of Features >> {ds.shape[1]}')
    print(f'Original >> {ds.shape[0]} Samples')
    ds_samples = ds.shape[0]
    nd = ds.drop_duplicates()
    print(f'No Duplicate >> {nd.shape[0]} Samples')
    nd_samples = nd.shape[0]
    percent = 1.0 - (nd_samples/ds_samples)
    percent *= 100.0
    print(f'Reduction >> {percent:.2f}%')

def frequency_features(dataset, args):
    """
    Avalia a frequência das características em um conjunto de dados e calcula a porcentagem máxima de ocorrência.

    Parâmetros:
    - dataset : DataFrame contendo o conjunto de dados a ser analisado, incluindo as características e, possivelmente, a coluna de classe.
    - args : Objeto contendo os argumentos de entrada que definem parâmetros como a coluna de classe.
    """
    print_info('>>> Evaluating Frequency of Features <<<')
    ft_sum = dataset.sum()
    ft_sum.drop(labels = [args.class_column], inplace = True)
    data = {'feature': list(ft_sum.index), 'frequency': ft_sum.values}
    fdf = pd.DataFrame(data)
    #fdf.sort_values(by = ['frequency'], ascending = False, inplace = True)
    print(f'Max. Feature Frequency >> {ft_sum.max()} ({(ft_sum.max()/len(dataset)) * 100.0:.2f}%)')
    print_distribution(fdf, 'frequency', 0.1, len(dataset))

def calculate_rf_features_importance(features, target):
    """
    Calcula a importância das características utilizando um classificador Random Forest e retorna um DataFrame com as características mais importantes.

    Parâmetros:
    - features :  DataFrame contendo as características (features) do conjunto de dados, onde cada coluna representa uma característica.
    - target : Série contendo os rótulos de classe ou variáveis alvo associadas às amostras em `features`.

    Retorna:
    - df : DataFrame
        DataFrame contendo a importância das características calculada pelo classificador Random Forest. O DataFrame possui duas colunas:
        - "feature": Nome das características.
        - "importance": Importância associada a cada característica.
    """
    features_names = features.columns
    model = RandomForestClassifier(random_state = 0)
    model.fit(features, target)
    #print('>>>>', model.score(features, target))
    importance = model.feature_importances_
    data = {'feature': features_names, 'importance': importance}
    df = pd.DataFrame(data)
    df = df.sort_values(by = ['importance'], ascending = False)
    return df

def rf_features_importance(dataset, args):
    """
    Avalia a importância das características usando um classificador Random Forest e imprime a importância máxima e a distribuição das importâncias das características.

    Parâmetros:
    - dataset :DataFrame contendo o conjunto de dados. A última coluna é assumida como a coluna de classe (rótulos), enquanto as demais são características.
    - args : Objeto contendo os argumentos de linha de comando, incluindo o nome da coluna de classe e uma opção para verbosidade.
    """
    print_info('>>> Evaluating RF Features Importance <<<')
    X_dataset, y_dataset = get_X_y(args, dataset)
    rffi = calculate_rf_features_importance(X_dataset, y_dataset)
    max_rffi = rffi['importance'].max()
    print(f'Max. RF Feature Importance >> {max_rffi:.5f}')
    if args.verbose:
        print_distribution(rffi, 'importance', 0.1)

def correlation_coefficient(dataset, args):
    """
    Avalia o coeficiente de correlação entre características no conjunto de dados e imprime informações sobre pares de características com alta e baixa correlação.

    Parâmetros:
    - dataset :DataFrame contendo o conjunto de dados. A última coluna é assumida como a coluna de classe (rótulos), enquanto as demais são características.
    - args : Objeto contendo os argumentos de linha de comando, incluindo o nome da coluna de classe e uma opção para verbosidade.
    """
    print_info('>>> Evaluating Correlation Coefficient <<<')
    X_dataset, y_dataset = get_X_y(args, dataset)
    correlation = X_dataset.corr(method = 'kendall')
    features_pairs = list()
    corr_values = list()
    pairs_count = 0
    low_pairs_count = 0
    for index in correlation.index:
        for column in correlation.columns:
            if index == column:
                break
            pairs_count += 1
            if abs(correlation.loc[index, column]) > 0.8:
                p = sorted([index, column])
                if p not in features_pairs:
                    features_pairs.append(p)
                    corr_values.append(correlation.loc[index, column])
            if abs(correlation.loc[index, column]) < 0.2:
                low_pairs_count += 1
    data = {'pairs': features_pairs, 'corr': corr_values}
    df = pd.DataFrame(data)
    df = df.sort_values(by = ['corr'], ascending = False)
    percent = (len(df)/pairs_count) * 100.0
    print(f'Number of Pairs With High Correlation (> 0.8): {percent:.2f}% ({len(df)})')
    percent = (low_pairs_count/pairs_count) * 100.0
    print(f'Number of Pairs With Low Correlation (< 0.2): {percent:.2f}% ({low_pairs_count})')

    if args.verbose and len(df):
        print(df)

def vif(dataset, args):
    """
    Avalia o Fator de Inflação da Variância (VIF) para cada característica no conjunto de dados e imprime informações sobre características com alta colinearidade.

    Parâmetros:
    - dataset : DataFrame contendo o conjunto de dados. A última coluna é assumida como a coluna de classe (rótulos), enquanto as demais são características.
    - args : Objeto contendo os argumentos de linha de comando, incluindo o nome da coluna de classe e uma opção para verbosidade.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    print_info('>>> Evaluating Variance Inflation Factor (VIF) <<<')
    X_dataset, y_dataset = get_X_y(args, dataset)
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_dataset.columns
    vif_data["vif"] = [variance_inflation_factor(X_dataset.values, i) for i in range(len(X_dataset.columns))]
    high_vif = vif_data[vif_data['vif'] > 5.0]
    high_vif = high_vif.sort_values(by = ['vif'], ascending = False)
    percent = (len(high_vif)/len(vif_data)) * 100.0
    max_vif = vif_data['vif'].max()
    print(f'Max. VIF >> {max_vif:.2f}')
    print(f'Number of Features With High VIF (> 5.0): {percent:.2f}% ({len(high_vif)})')
    if args.verbose and len(high_vif):
        print(high_vif)

def calculate_permutation_features_importance(features, target):
    """
    Calcula a importância das características utilizando a técnica de importância por permutação e retorna as características mais importantes com base na importância calculada.

    Parâmetros:
    - features :DataFrame contendo as características do conjunto de dados, onde cada coluna representa uma característica.
    - target : Série contendo os rótulos de classe ou variáveis alvo associadas às amostras em `features`.

    Retorna:
    - df : DataFrame
        DataFrame contendo as características ordenadas pela importância calculada por permutação. O DataFrame possui duas colunas:
            - "feature": Nome das características.
            - "importance": Importância calculada para cada característica.
    """
    from sklearn.inspection import permutation_importance
    features_names = features.columns
    model = RandomForestClassifier(random_state = 0)
    model.fit(features, target)
    #print('>>>>', model.score(features, target))
    results = permutation_importance(model, features, target, scoring='accuracy')
    importance = results.importances_mean
    data = {'feature': features_names, 'importance': importance}
    df = pd.DataFrame(data)
    df.sort_values(by = ['importance'], ascending = False, inplace = True)
    return df

def permutation_features_importance(dataset, args):
    """
    Avalia a importância das características utilizando a técnica de importância por permutação e exibe os resultados.

    Parâmetros:
    - dataset : DataFrame contendo o conjunto de dados completo, incluindo características e coluna de classe.
    - args :  Objeto contendo argumentos e opções, incluindo o nome da coluna de classe e se a saída detalhada deve ser exibida.
    """
    print_info('>>> Evaluating Permutation Features Importance <<<')
    X_dataset, y_dataset = get_X_y(args, dataset)
    pfi = calculate_permutation_features_importance(X_dataset, y_dataset)
    max_pfi = pfi['importance'].max()
    print(f'Max. Permutation Feature Importance >> {max_pfi:.5f}')
    if args.verbose:
        print_distribution(pfi, 'importance', 0.1)

def print_distribution(df, metric, step, max_value = 1.0):
    """
    Imprime a distribuição das características com base em um critério métrico em diferentes níveis de limiar.

    Parâmetros:
    - df :  DataFrame contendo as características e suas respectivas métricas. Deve ter pelo menos duas colunas: uma para a métrica e uma para o nome da característica.
    - metric : Nome da coluna no DataFrame que contém os valores das métricas para as características.
    - step : O incremento do limiar para a análise da distribuição. Por exemplo, um passo de 0.1 resulta em limiares de 0.0, 0.1, 0.2, ..., 1.0.
    - max_value : O valor máximo da métrica utilizado para calcular os limiares. Valor padrão é 1.0;
    """
    pd.options.display.float_format = '{:.2f}'.format
    df.style.hide_index()
    count = 0
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
        if afc == 100.0:
            break
    data_df = pd.DataFrame(data_list, columns = ['threshold (%)', 'relative (%)', 'cumulative (%)', 'absolute'])
    print(data_df.to_string(index = False))

if __name__=="__main__":
    args = parse_args(sys.argv[1:])

    try:
        dataset = pd.read_csv(args.dataset)
    except BaseException as e:
        print('Exception: {}'.format(e))
        exit(1)

    duplicate_samples(dataset, args)
    frequency_features(dataset, args)
    mig_features(dataset, args)
    rf_features_importance(dataset, args)
    permutation_features_importance(dataset, args)
    correlation_coefficient(dataset, args)
    vif(dataset, args)

    '''
    dq = DataQuality(df = dataset)
    x = dq.get_warnings(test="Duplicate Columns")
    print(x)
    results = dq.evaluate()
    print(results)
    '''
