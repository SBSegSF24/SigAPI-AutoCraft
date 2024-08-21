import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import sys
from random import choice
from argparse import ArgumentParser
from SigAPI_Otimizado.metodos.utils import get_base_parser, get_dataset, get_X_y
import matplotlib.pyplot as plt
from sklearn import metrics

def correlation_phase(X, y, k, method, methods):
    """
    Executa a fase de correlação em um conjunto de dados, removendo características altamente correlacionadas.

    Parâmetros:
    - X : DataFrame contendo as características (features) do conjunto de dados.
    - y : Série contendo os rótulos de classe correspondentes às amostras em `X`.
    - k : Número de características a serem selecionadas pelo método de seleção de características especificado.
    - method : Nome do método de seleção de características a ser utilizado. Este nome deve corresponder a uma chave no dicionário `methods`.
    - methods : Dicionário onde as chaves são nomes de métodos de seleção de características e os valores são dicionários que contêm as funções de seleção associadas.

    Retorna:
    - new_X : DataFrame contendo as características após a remoção de características altamente correlacionadas, com a coluna de rótulos de classe adicionada de volta.
    """



    feature_scores = methods[method]['function'](X, y, k)
    new_X = X[list(feature_scores['features'])]

    correlation = new_X.corr()

    model_RF=RandomForestClassifier()
    model_RF.fit(new_X,y)

    feats = {}
    for feature, importance in zip(new_X.columns, model_RF.feature_importances_):
        feats[feature] = importance

    to_drop = set()

    for index in correlation.index:
        for column in correlation.columns:
            if index != column and correlation.loc[index, column] > 0.85:
               ft = column if feats[column] <= feats[index] else index
               to_drop.add(ft)
    print("qtd de features removidas:", len(to_drop))

    new_X = new_X.drop(columns = to_drop)
    new_X['class'] = y
    return new_X

def parse_args(argv):
    """
    Analisa e retorna os argumentos da linha de comando fornecidos ao script.

    Parâmetros:
    - argv : Lista de argumentos da linha de comando.

    Retorna:
    - args : Objeto Namespace contendo os valores dos argumentos de linha de comando, com os seguintes atributos:
        - threshold : Threshold para a diferença entre métricas em cada incremento no número de características, valor padrão: 0.03.
        - initial_n_features : Número inicial de características a serem utilizadas, valor padrão: 1.
        - increment : Valor pelo qual o número inicial de características será incrementado em cada iteração, valor padrão: 1.
    """    


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

def get_moving_average(data, window_size=5):
    """
    Calcula a média móvel de uma série  utilizando uma janela deslizante.

    Parâmetros:
    - data (: Sequência de dados numéricos sobre a qual a média móvel será calculada.
    - window_size : Tamanho da janela deslizante utilizada para calcular a média móvel. O valor padrão é 5.

    Retorna:
    - moving_averages : Um array contendo as médias móveis calculadas para a série de dados, de acordo com o tamanho da janela especificado.
    """    
    cumsum_vec = np.cumsum(np.insert(data, 0, 0))
    return (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size

def get_minimal_range_suggestion(df, t=0.001, window_size=5):
    """
    Sugere o menor índice no qual a variação de gradiente de médias móveis excede um determinado limiar, dentro de uma janela deslizante.

    Parâmetros:
    - df : DataFrame contendo os dados numéricos sobre os quais a análise será realizada.
    - t : Limiar (threshold) para a diferença de gradiente que deve ser excedido para considerar uma mudança significativa. O valor padrão é 0.001.
    - window_size (: Tamanho da janela deslizante utilizada para calcular as médias móveis. O valor padrão é 5.

    Retorna:
    - int: O menor índice no DataFrame onde a variação no gradiente da média móvel excede o limiar `t`. Se nenhuma variação satisfatória for encontrada, retorna `-1`.
    """

    moving_averages = np.array([get_moving_average(np.array(df)[:, i], window_size) for i in range(df.shape[1])]).T
    gradients = np.gradient(moving_averages, axis=0)
    diffs = gradients[1:] - gradients[:-1]

    for i in range(len(diffs) - 1, 1, -1):
        if(any([diff > t for diff in diffs[i]])):
            return int(df.index[i])
    return -1

"""# **Função Incremento** """

def calculateMutualInformationGain(features, target, k):
    """
    Calcula o ganho de informação mútua para as características de um conjunto de dados e retorna as principais características baseadas nesse ganho.

    Parâmetros:
    - features : DataFrame contendo as características (features) do conjunto de dados, onde cada coluna representa uma característica.
    - target : Série contendo os rótulos de classe ou variáveis alvo associadas às amostras em `features`.
    - k : Número de principais características a serem retornadas com base no ganho de informação mútua.

    Retorna:
    - df: DataFrame contendo as `k` características com o maior ganho de informação mútua, ordenado em ordem decrescente de ganho. O DataFrame possui duas colunas:
        - "features": Nome das características.
        - "score": Ganho de informação mútua associado a cada característica.
    """
    feature_names = features.columns
    mutualInformationGain = mutual_info_classif(features, target, random_state = 0)
    data = {"features": feature_names, "score": mutualInformationGain}
    df = pd.DataFrame(data)
    df = df.sort_values(by=['score'], ascending=False)
    return df[:k]

def calculateRandomForestClassifier(features, target,k):
    """
    Calcula a importância das características utilizando um classificador Random Forest e retorna as principais características com base na importância.

    Parâmetros:
    - features : DataFrame contendo as características (features) do conjunto de dados, onde cada coluna representa uma característica.
    - target : Série contendo os rótulos de classe ou variáveis alvo associadas às amostras em `features`.
    - k : Número de principais características a serem retornadas com base no ganho de informação mútua.

    Retorna:
    - df: DataFrame contendo as `k` características com o maior ganho de informação mútua, ordenado em ordem decrescente de ganho. O DataFrame possui duas colunas:
        - "features": Nome das características.
        - "score": Ganho de informação mútua associado a cada característica.
    """   
    feature_names= np.array(features.columns.values.tolist())
    test = RandomForestClassifier(random_state = 0)
    test = test.fit(features,target)
    model = SelectFromModel(test,max_features = k, prefit = True)
    model.get_support()
    best_features = feature_names[model.get_support()]
    best_score = test.feature_importances_[model.get_support()]
    df = pd.DataFrame(list(zip(best_features,best_score)),columns=['features','score']).sort_values(by= ['score'],ascending= False)
    return df

def calculateExtraTreesClassifier(features, target, k):
    """
    Calcula a importância das características utilizando um classificador Extra Trees e retorna as principais características com base na importância.

    Parâmetros:
    - features : DataFrame contendo as características (features) do conjunto de dados, onde cada coluna representa uma característica.
    - target : Série contendo os rótulos de classe ou variáveis alvo associadas às amostras em `features`.
    - k : Número de principais características a serem retornadas com base no ganho de informação mútua.

    Retorna:
    - df: DataFrame contendo as `k` características com o maior ganho de informação mútua, ordenado em ordem decrescente de ganho. O DataFrame possui duas colunas:
        - "features": Nome das características.
        - "score": Ganho de informação mútua associado a cada característica.
    """
    feature_names= np.array(features.columns.values.tolist())
    test = ExtraTreesClassifier(random_state = 0)
    test = test.fit(features,target)
    model = SelectFromModel(test,max_features = k, prefit = True)
    model.get_support()
    best_features = feature_names[model.get_support()]
    best_score = test.feature_importances_[model.get_support()]
    df = pd.DataFrame(list(zip(best_features,best_score)),columns=['features','score']).sort_values(by= ['score'],ascending= False)
    return df

def calculateRFERandomForestClassifier(features, target, k):
    """
    Calcula a importância das características utilizando um classificador RFE Random Forest e retorna as principais características com base na importância.

    Parâmetros:
    - features : DataFrame contendo as características (features) do conjunto de dados, onde cada coluna representa uma característica.
    - target : Série contendo os rótulos de classe ou variáveis alvo associadas às amostras em `features`.
    - k : Número de principais características a serem retornadas com base no ganho de informação mútua.

    Retorna:
    - df: DataFrame contendo as `k` características com o maior ganho de informação mútua, ordenado em ordem decrescente de ganho. O DataFrame possui duas colunas:
        - "features": Nome das características.
        - "score": Ganho de informação mútua associado a cada característica.
    """

    feature_names= np.array(features.columns.values.tolist())
    rfe = RFE(estimator = RandomForestClassifier(), n_features_to_select = k)
    model = rfe.fit(features,target)
    best_features = feature_names[model.support_]
    best_scores = rfe.estimator_.feature_importances_
    df = pd.DataFrame(list(zip(best_features, best_scores)), columns = ["features", "score"]).sort_values(by = ['score'], ascending=False)
    return df

def calculateRFEGradientBoostingClassifier(features, target,k):
    """
    Calcula a importância das características utilizando um classificador RFE Gradient Boost e retorna as principais características com base na importância.

    Parâmetros:
    - features : DataFrame contendo as características (features) do conjunto de dados, onde cada coluna representa uma característica.
    - target : Série contendo os rótulos de classe ou variáveis alvo associadas às amostras em `features`.
    - k : Número máximo de principais características a serem retornadas com base na importância.

    Retorna:
    - df: DataFrame contendo as `k` características mais importantes com base na importância calculada pelo classificador RFE Gradient Boost. O DataFrame possui duas colunas:
        - "features": Nome das características.
        - "score": Importância associada a cada característica.
    """

    feature_names= np.array(features.columns.values.tolist())
    rfe = RFE(estimator = GradientBoostingClassifier(), n_features_to_select = k)
    model = rfe.fit(features, target)
    best_features = feature_names[model.support_]
    best_scores = rfe.estimator_.feature_importances_
    df = pd.DataFrame(list(zip(best_features, best_scores)), columns = ["features", "score"]).sort_values(by = ['score'], ascending=False)
    return df


def calculateSelectKBest(features, target,k):
    """
    Calcula a pontuação de seleção de características usando o método Chi-quadrado e retorna as principais características com base nessas pontuações.

    Parâmetros:
    - features : DataFrame contendo as características (features) do conjunto de dados, onde cada coluna representa uma característica.
    - target : Série contendo os rótulos de classe ou variáveis alvo associadas às amostras em `features`.
    - k : Número de principais características a serem retornadas com base no ganho de informação mútua.

    Retorna:
    - df: DataFrame contendo as `k` características com o maior ganho de informação mútua, ordenado em ordem decrescente de ganho. O DataFrame possui duas colunas:
        - "features": Nome das características.
        - "score": Ganho de informação mútua associado a cada característica.
    """   
    feature_names= np.array(features.columns.values.tolist())
    chi2_selector= SelectKBest(score_func = chi2, k= k)
    chi2_selector.fit(features,target)
    chi2_scores = pd.DataFrame(list(zip(feature_names,chi2_selector.scores_)),columns= ['features','score'])
    df = pd.DataFrame(list(zip(feature_names,chi2_selector.scores_)),columns= ['features','score']).sort_values(by = ['score'], ascending=False)
    return df[:k]


def calculateMetricas(new_X,y):
    """
    Calcula as métricas de desempenho de um classificador Random Forest em um conjunto de dados de teste e retorna uma lista com os resultados das métricas.

    Parâmetros:
    - new_X : DataFrame contendo as características (features) para treinamento e teste do classificador.
    - y : Série contendo os rótulos de classe ou variáveis alvo associadas às amostras em `new_X`.

    Retorna:
    - metricas: Lista contendo as métricas de desempenho calculadas. A lista inclui os seguintes valores na ordem:
        - Acurácia (accuracy)
        - Precisão (precision)
        - Revocação (recall)
        - F1 Score (f1)
    """    
    new_X_train,new_X_test, y_train, y_test = train_test_split(new_X, y, test_size = 0.3,random_state = 0)

    teste = RandomForestClassifier()
    teste.fit(new_X_train, y_train)
    resultado_teste = teste.predict(new_X_test)

    acuracia = accuracy_score(y_test, resultado_teste)
    precision = precision_score(y_test, resultado_teste, zero_division = 0)
    recall = recall_score(y_test, resultado_teste, zero_division = 0)
    f1 = f1_score(y_test, resultado_teste, zero_division = 0)

    metricas = [acuracia,precision,recall,f1]
    return metricas

methods = { 'mutualInformation': { 'function': calculateMutualInformationGain, 'results': [[0,0,0,0,0]], 'is_stable': False },
    'selectRandom': { 'function': calculateRandomForestClassifier, 'results': [[0,0,0,0,0]], 'is_stable': False },
    'selectExtra': { 'function': calculateExtraTreesClassifier, 'results': [[0,0,0,0,0]], 'is_stable': False },
    'RFERandom': { 'function': calculateRFERandomForestClassifier, 'results': [[0,0,0,0,0]], 'is_stable': False },
    'RFEGradient': { 'function': calculateRFEGradientBoostingClassifier, 'results': [[0,0,0,0,0]], 'is_stable': False },
    'selectKBest': { 'function': calculateSelectKBest, 'results': [[0,0,0,0,0]], 'is_stable': False }
}
#---------------------------------------------------------------------

def distribution(df, metric, step, max_value = 1.0):
    """
    Calcula e analisa a distribuição de uma métrica em um DataFrame, baseada em thresholds variáveis. A função também determina o número mínimo de características selecionadas para um dado ponto percentual na distribuição.

    Parâmetros:
    - df : DataFrame contendo as características e suas respectivas métricas. Deve incluir a coluna referenciada por `metric`.
    - metric : Nome da coluna no DataFrame `df` que contém as métricas a serem analisadas.
    - step : O tamanho do passo para variar o threshold. Define a precisão da análise percentual.
    - max_value : O valor máximo do threshold a ser considerado. Valor padrão é 1.0.

    Retorna:
    - min_features: Número mínimo de características selecionadas para um threshold percentual de pelo menos 30%. Este valor representa a soma acumulada das características selecionadas quando o percentual de threshold atinge ou excede 30%.
    """
    pd.options.display.float_format = '{:.2f}'.format
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
    print("Min. Features:", min_features)
    data_df = pd.DataFrame(data_list, columns = ['threshold (%)', 'relative (%)', 'cumulative (%)', 'absolute'])
    return min_features
    #print(data_df.to_string(index = False))


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

def mig_features():
    """
    Avalia o ganho de informação mútua (Mutual Information Gain) das características em um conjunto de dados, imprime o valor máximo de ganho de informação e calcula a distribuição das características com base em um limiar definido.

    Esta função utiliza as seguintes funções auxiliares:
    - `get_X_y`: Extrai as características e o alvo do conjunto de dados com base nos argumentos fornecidos.
    - `calculate_mig_features`:  Calcula o ganho de informação mútua (Mutual Information Gain) para as características fornecidas e retorna um DataFrame contendo a importância de cada característica.
    - `distribution`: Calcula e analisa a distribuição de uma métrica em um DataFrame, baseada em thresholds variáveis. A função também determina o número mínimo de características selecionadas para um dado ponto percentual na distribuição.


    Retorna:
    - min: O número mínimo de características com base na distribuição do ganho de informação mútua.
    """   
    #print_info('>>> Evaluating Information Gain <<<')
    X_dataset, y_dataset = get_X_y(parsed_args, get_dataset(parsed_args))
    mig = calculate_mig_features(X_dataset, y_dataset)
    max_ig = mig['score'].max()
    print(f'Max. IG {max_ig:.5f}')
    min = distribution(mig, 'score', 0.1, max_ig)
    return min

def pca_features():  
    """
    Avalia a importância das características utilizando Análise de Componentes Principais (PCA), imprime o valor máximo da importância das componentes principais e calcula a distribuição das características com base em um limiar definido.

    Esta função utiliza as seguintes funções auxiliares:
    - `get_X_y`: Extrai as características e o alvo do conjunto de dados com base nos argumentos fornecidos.
    - `calculate_PCA`: Calcula a importância das características utilizando Análise de Componentes Principais (PCA). Retorna um DataFrame com as características ordenadas pela importância média calculada a partir das componentes principais.
    - `distribution`: Calcula e analisa a distribuição de uma métrica em um DataFrame, baseada em thresholds variáveis. A função também determina o número mínimo de características selecionadas para um dado ponto percentual na distribuição.

    Retorna:
    - min: O número mínimo de características com base na distribuição do ganho de informação mútua.
    """
    #print_info('>>> PCA <<<')
    X_dataset, y_dataset = get_X_y(parsed_args, get_dataset(parsed_args))
    pca = calculate_PCA(X_dataset)
    max_PCA = pca['score'].max()  
    print(f'Max. PCA {max_PCA:.5f}')
    min = distribution(pca, 'score', 0.1, max_PCA)
    return min

def calculate_PCA(features):
    """
    Calcula a importância das características utilizando Análise de Componentes Principais (PCA). Retorna um DataFrame com as características ordenadas pela importância média calculada a partir das componentes principais.

    Parâmetros:
    - features : DataFrame
        DataFrame contendo as características (features) do conjunto de dados, onde cada coluna representa uma característica.

    Retorna:
    - df
        DataFrame contendo as características ordenadas pela importância calculada. O DataFrame possui duas colunas:
        - "feature": Nome das características.
        - "score": Importância média associada a cada característica.
    """
    features_names = features.columns
    n_components = 3
    pca = PCA(n_components)
    pca.fit(features)
    
    score = pca.components_.T * np.sqrt(pca.explained_variance_)[np.newaxis, :]
    data = {'feature': features_names, 'score': np.abs(score).mean(axis=1)}  # Taking mean of absolute values
    df = pd.DataFrame(data)
    df = df.sort_values(by=['score'], ascending=False)
    return df
#---------------------------------------------------------------------

def is_method_stable(previous_metrics, current_metrics, t=0.03):
    """
    Verifica se as métricas atuais de desempenho de um modelo são estáveis em relação às métricas anteriores, com base em um limiar de tolerância.

    Parâmetros:
    - previous_metrics : Lista ou array contendo as métricas de desempenho anteriores do modelo.
    - current_metrics : Lista ou array contendo as métricas de desempenho atuais do modelo.
    - t : Tolerância para a diferença entre as métricas anteriores e atuais. Valor padrão: 0.03.

    Retorna:
    - bool: `True` se todas as diferenças absolutas entre as métricas atuais e anteriores forem menores que o limiar `t`, indicando que o método é estável. Caso contrário, retorna `False`.
    """
    differences = abs(current_metrics - previous_metrics)
    print("differences: ",differences)
    if(all(differences < t)):
        return True
    return False

def calculate_distance(accuracy, precision, recall, f1):
    """
    Calcula a distância euclidiana média entre as métricas de desempenho fornecidas (acurácia, precisão, recall e F1-score).

    Parâmetros:
    - accuracy : Valor da métrica de acurácia (accuracy), representando a proporção de previsões corretas sobre o total de previsões.
    - precision :  Valor da métrica de precisão (precision), representando a proporção de verdadeiros positivos sobre o total de previsões positivas.
    - recall :  Valor da métrica de recall (recall), representando a proporção de verdadeiros positivos sobre o total de casos positivos reais.
    - f1 : Valor da métrica F1-score, a média harmônica ponderada entre precisão e recall.

    Retorna:
    - distance: Distância euclidiana média calculada a partir das métricas fornecidas.
    """
    # calcular a distância entre as métricas
    # distância euclidiana média
    distance = np.mean(np.linalg.norm(np.array([accuracy, precision, recall, f1]), axis=0))
    return distance

def calculate_triangle_area(base, height):
    return 0.5 * base * height

def plot_radar_with_area(metrics_history):
    """
    Plota gráficos de radar e calcula a área total coberta pelas métricas de desempenho para diferentes métodos.

    Parâmetros:
    - metrics_history : dict
        Dicionário onde as chaves são os nomes dos métodos e os valores são dicionários contendo listas de métricas ao longo das iterações. Cada dicionário de métricas possui as seguintes chaves:
        - 'features' : Lista de números de características usadas em cada iteração.
        - 'accuracy' : Lista de valores de acurácia (accuracy) calculados ao longo das iterações.
        - 'precision' : 
            Lista de valores de precisão (precision) calculados ao longo das iterações.
        - 'recall' : list
            Lista de valores de recall (recall) calculados ao longo das iterações.
        - 'f1' : list
            Lista de valores de F1-score calculados ao longo das iterações.

    Retorna:
    - float
        Área total coberta pelas métricas de desempenho para todos os métodos.
    """
    total_area = 0

    for method_name, metrics in metrics_history.items():
        # Calcular as áreas dos triângulos para cada métrica
        area_accuracy = calculate_triangle_area(1, metrics['accuracy'][-1])
        area_precision = calculate_triangle_area(1, metrics['precision'][-1])
        area_recall = calculate_triangle_area(1, metrics['recall'][-1])
        area_f1 = calculate_triangle_area(1, metrics['f1'][-1])

        # Calcular a área total
        total_area = area_accuracy + area_precision + area_recall + area_f1
    return total_area

    # Plotar o gráfico de radar
    '''categories = ['Accuracy', 'Precision', 'Recall', 'F1']
    values = [area_accuracy, area_precision, area_recall, area_f1]
    angles = [n / float(len(categories)) * 2 * 3.14159 for n in range(len(categories))]
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='blue', alpha=0.25)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.yaxis.grid(True)

    plt.title(f'Total Area: {total_area}', size=20, color='blue', y=1.1)
    plt.show()'''

def selection_phase(X, y, cols, methods, num_features=1, increment=1):
    """
    Realiza a fase de seleção de características utilizando diversos métodos e avalia o desempenho dos métodos ao longo do processo de seleção. 
    - X : DataFrame contendo as características (features) do conjunto de dados.
    - y ( Série contendo os rótulos de classe ou variáveis alvo associadas às amostras em `X`.
    - methods : Dicionário contendo os métodos de seleção de características e suas respectivas funções e resultados. A estrutura esperada é:
    - Chave : Nome do método.
    - Valor : Dicionário contendo:
        - 'function': Função que realiza a seleção de características e retorna um DataFrame com as características e suas pontuações.
        - 'results': Array ou lista para armazenar os resultados da métrica para cada número de características.
    - num_features : Número inicial de características a serem consideradas. Valor padrão: 1.
    - increment : Valor para incrementar o número de características a cada iteração. Valor padrão: 1.

    Retorna:
        - best_stable_method : Nome do método de seleção de características que apresentou a maior acurácia estável.
        - k : Número de características associadas ao melhor método estável.
    """
    has_found_stable_method = False
    best_stable_method = None
    #best_distance = float('inf')
    best_area = 0
    best_mediana = 0

    # Adicionando para armazenar as métricas ao longo do processo
    metrics_history = {method_name: {'features': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []} for method_name in methods.keys()}

    while num_features < (total_features + increment) and not has_found_stable_method:
        k = total_features if num_features > total_features else num_features
        print("qtd de features: ", k)

        for method_name in methods.keys():
            feature_scores = methods[method_name]['function'](X, y, k)
            new_X = X[list(feature_scores['features'])]
            metrics =  calculateMetricas(new_X,y)
            methods[method_name]['results'] = np.append(methods[method_name]['results'],[[k,metrics[0],metrics[1],metrics[2],metrics[3]]],axis=0)
            previous_metrics = methods[method_name]['results'][-2][1:]
            current_metrics = methods[method_name]['results'][-1][1:]

            # Adicionando métricas ao histórico
            metrics_history[method_name]['features'].append(k)
            metrics_history[method_name]['accuracy'].append(metrics[0])
            metrics_history[method_name]['precision'].append(metrics[1])
            metrics_history[method_name]['recall'].append(metrics[2])
            metrics_history[method_name]['f1'].append(metrics[3])

            # A primeira expressão booleana (len(...) > 2) é para evitar comparar as métricas calculadas contra o vetor [0,0,0,0],
            # que é definido inicialmente no dicionário "methods"
            print("Metodos: ", method_name, "Len: ", len(methods[method_name]['results']), "Results: ",  methods[method_name]['results'])
            if len(methods[method_name]['results']) > cols and is_method_stable(previous_metrics, current_metrics, parsed_args.threshold):

                #if ('''recall >= RF_recall and precision >= RF_precision and accuracy >= RF_accuracy and f1 >= RF_f1'''):
                has_found_stable_method = True
                accuracy = current_metrics[0]
                precision = current_metrics[1]
                recall = current_metrics[2]
                f1 = current_metrics[3]

                lista = [accuracy, precision, recall, f1]
                print('Lista Desordenada: ', lista)
                lista_ordenada = sorted(lista)
                print('Lista Ordenada: ', lista_ordenada)

                mediana = (lista_ordenada[1]+lista_ordenada[2])/2
                #distance = calculate_distance(metrics_history[method_name]['accuracy'], metrics_history[method_name]['precision'], metrics_history[method_name]['recall'], metrics_history[method_name]['f1'])
                if mediana > best_mediana:
                    best_mediana = mediana
                    print(mediana)
                    best_stable_method = method_name

        num_features += increment
            

    if(not has_found_stable_method):
        best_stable_method = choice(list(methods.keys()))

    k = int(methods[best_stable_method]["results"][-1][0])

    return best_stable_method, k

if __name__=="__main__":
    start = time.time()
    parsed_args = parse_args(sys.argv[1:])
    X, y = get_X_y(parsed_args, get_dataset(parsed_args))
    total_features = get_dataset(parsed_args).shape[1] - 1
    print(f'Number of Features >> {X.shape[1]}')

    #metricas_RandomForest = calculateRF(get_dataset(parsed_args));

    if(parsed_args.initial_n_features > total_features):
        print(f"ERRO: --initial-n-features ({parsed_args.initial_n_features}) maior que a qtd de features do dataset ({total_features})")
        exit(1)
    
    result = (15*total_features)/100
    print('15% do dataset - Result: ', int(result))

    cols = mig_features()
    print('Cols IG (Threshold 30%)', cols)
    if (cols > (2*int(result))):
        cols = pca_features()
        print('Cols PCA (Threshold 30%)', cols)
        if (cols < int(result)) or (cols > (2*int(result))):
            cols = int(result)
            print('Cols 15%: ', int(result))
    if cols < int(result):
        cols = pca_features()
        print('Cols PCA (Threshold 30%)', cols)
        if (cols < int(result)) or (cols > (2*int(result))):
            cols = int(result)
            print('Cols 15%: ', int(result))

    '''result = (15*total_features)/100
    print('15 pct do dataset - Result: ', int(result))

    cols = mig_features(30.0)
    print('Cols IG (Threshold 30%)', cols)
    if cols <int(result):
        cols = mig_features(20.0)
        print('Cols IG (Threshold 20%)', cols)

    if cols < int(result):
        cols = pca_features(30.0)
        print('Cols PCA (Threshold 30%)', cols)
        if cols < int(result):
            cols = pca_features(20.0)
            print('Cols PCA (Threshold 20%)', cols)'''

    best_stable_method, lower_bound = selection_phase(X, y, cols, methods, num_features=parsed_args.initial_n_features, increment=parsed_args.increment)
    print(f'Menor limite inferior encontrado: {best_stable_method}, {lower_bound}')

    new_X = correlation_phase(X, y, lower_bound, best_stable_method, methods)
    new_X.to_csv(parsed_args.output_file, index=False)
    print("Dataset final criado")

    stop = time.time()
    total_time = stop - start
    print(f'Time: {total_time: .2f} segundos.')