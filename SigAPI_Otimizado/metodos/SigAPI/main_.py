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
    base_parser = get_base_parser()
    parser = ArgumentParser(parents=[base_parser])
    parser.add_argument('-t', '--threshold', type = float, default = 0.03,
        help = 'Threshold for the difference between metrics at each increment on the number of features. When all metrics are less than it, the selection phase finishes. Default: 0.03')
    parser.add_argument( '-f', '--initial-n-features', type = int, default = 1,
        help = 'Initial number of features. Default: 1')
    parser.add_argument( '-i', '--increment', type = int, default = 1,
        help = 'Value to increment the initial number of features. Default: 1')
    parser.add_argument('-met', '--metrics-output-file', type=str, required=True,
        help='File path to save the metrics.')
    args = parser.parse_args(argv)
    return args

def get_moving_average(data, window_size=5):
    cumsum_vec = np.cumsum(np.insert(data, 0, 0))
    return (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size

def get_minimal_range_suggestion(df, t=0.001, window_size=5):
    moving_averages = np.array([get_moving_average(np.array(df)[:, i], window_size) for i in range(df.shape[1])]).T
    gradients = np.gradient(moving_averages, axis=0)
    diffs = gradients[1:] - gradients[:-1]

    for i in range(len(diffs) - 1, 1, -1):
        if(any([diff > t for diff in diffs[i]])):
            return int(df.index[i])
    return -1

"""# **Função Incremento** """

def calculateMutualInformationGain(features, target, k):
    feature_names = features.columns
    mutualInformationGain = mutual_info_classif(features, target, random_state = 0)
    data = {"features": feature_names, "score": mutualInformationGain}
    df = pd.DataFrame(data)
    df = df.sort_values(by=['score'], ascending=False)
    return df[:k]

def calculateRandomForestClassifier(features, target,k):
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
    feature_names= np.array(features.columns.values.tolist())
    rfe = RFE(estimator = RandomForestClassifier(), n_features_to_select = k)
    model = rfe.fit(features,target)
    best_features = feature_names[model.support_]
    best_scores = rfe.estimator_.feature_importances_
    df = pd.DataFrame(list(zip(best_features, best_scores)), columns = ["features", "score"]).sort_values(by = ['score'], ascending=False)
    return df

def calculateRFEGradientBoostingClassifier(features, target,k):
    feature_names= np.array(features.columns.values.tolist())
    rfe = RFE(estimator = GradientBoostingClassifier(), n_features_to_select = k)
    model = rfe.fit(features, target)
    best_features = feature_names[model.support_]
    best_scores = rfe.estimator_.feature_importances_
    df = pd.DataFrame(list(zip(best_features, best_scores)), columns = ["features", "score"]).sort_values(by = ['score'], ascending=False)
    return df


def calculateSelectKBest(features, target,k):
    feature_names= np.array(features.columns.values.tolist())
    chi2_selector= SelectKBest(score_func = chi2, k= k)
    chi2_selector.fit(features,target)
    chi2_scores = pd.DataFrame(list(zip(feature_names,chi2_selector.scores_)),columns= ['features','score'])
    df = pd.DataFrame(list(zip(feature_names,chi2_selector.scores_)),columns= ['features','score']).sort_values(by = ['score'], ascending=False)
    return df[:k]


def calculateMetricas(new_X,y):
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

def calculateRF(dataset):
    
    X = dataset.iloc[:,:-1] # features
    y = dataset.iloc[:,-1] # class

    new_X_train,new_X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,random_state = 0)

    clf = RandomForestClassifier(random_state = 0)
    clf.fit(new_X_train, y_train)
    pred = clf.predict(new_X_test)
    
    accuracy = metrics.accuracy_score(y_test, pred)
    precision = metrics.precision_score(y_test, pred, zero_division = 0)
    recall = metrics.recall_score(y_test, pred, zero_division = 0)
    f1 = metrics.f1_score(y_test, pred, zero_division = 0)

    precision *= 100.0
    accuracy *= 100.0
    recall *= 100.0
    f1 *= 100.0

    data = [{'Accuracy':accuracy,'Precision':precision,'Recall':recall,'F1_Score':f1}]
    df = pd.DataFrame(data)
    print(df)
    metricasRF = [accuracy,precision,recall,f1]
    return metricasRF

#---------------------------------------------------------------------

def is_method_stable(previous_metrics, current_metrics, t=0.03):
    differences = abs(current_metrics - previous_metrics)
    print("differences: ",differences)
    if(all(differences < t)):
        return True
    return False

def selection_phase_Ok(X, y, methods, num_features=1, increment=1):
    has_found_stable_method = False
    best_stable_method = None
    best_metric_value = 0

    # Adicionando para armazenar as métricas ao longo do processo
    metrics_history = {method_name: {'features': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []} for method_name in methods.keys()}
    print("Num Features: ", total_features)
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
            if(len(methods[method_name]['results']) > 50 and is_method_stable(previous_metrics, current_metrics, parsed_args.threshold)):
                has_found_stable_method = True
                accuracy = current_metrics[0]
                if(accuracy > best_metric_value):
                    best_metric_value = accuracy
                    best_stable_method = method_name
        num_features += increment

    if(not has_found_stable_method):
        best_stable_method = choice(list(methods.keys()))

    k = int(methods[best_stable_method]["results"][-1][0])

    # Gerar gráficos para cada método
    for method_name in methods.keys():
        plt.figure()
        plt.plot(metrics_history[method_name]['features'], metrics_history[method_name]['accuracy'], label='Acurácia')
        plt.plot(metrics_history[method_name]['features'], metrics_history[method_name]['precision'], label='Precisão')
        plt.plot(metrics_history[method_name]['features'], metrics_history[method_name]['recall'], label='Recall')
        plt.plot(metrics_history[method_name]['features'], metrics_history[method_name]['f1'], label='F1')

        plt.xlabel('Número de features')
        plt.ylabel('Métricas')
        plt.title(f'Desempenho de {method_name} ao longo do processo de seleção')
        plt.legend()
        plt.show()

    return best_stable_method, k

def selection_phase(X, y, methods, num_features=1, increment=1):
    has_found_stable_method = False
    best_stable_method = None
    best_metric_value = 0

    metrics_history = {method_name: {'features': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []} for method_name in methods.keys()}
    print("Num Features: ", total_features)
    
    while num_features < (total_features + increment) and not has_found_stable_method:
        k = total_features if num_features > total_features else num_features
        print("qtd de features: ", k)

        for method_name in methods.keys():
            feature_scores = methods[method_name]['function'](X, y, k)
            new_X = X[list(feature_scores['features'])]
            metrics = calculateMetricas(new_X, y)
            methods[method_name]['results'] = np.append(methods[method_name]['results'], [[k, metrics[0], metrics[1], metrics[2], metrics[3]]], axis=0)
            previous_metrics = methods[method_name]['results'][-2][1:]
            current_metrics = methods[method_name]['results'][-1][1:]

            # Adicionar métricas ao histórico
            metrics_history[method_name]['features'].append(k)
            metrics_history[method_name]['accuracy'].append(metrics[0])
            metrics_history[method_name]['precision'].append(metrics[1])
            metrics_history[method_name]['recall'].append(metrics[2])
            metrics_history[method_name]['f1'].append(metrics[3])

            print("Metodos: ", method_name, "Len: ", len(methods[method_name]['results']), "Results: ",  methods[method_name]['results'])
            if len(methods[method_name]['results']) > 100 and is_method_stable(previous_metrics, current_metrics, parsed_args.threshold):
                has_found_stable_method = True
                accuracy = current_metrics[0]
                if accuracy > best_metric_value:
                    best_metric_value = accuracy
                    best_stable_method = method_name
        num_features += increment

    if not has_found_stable_method:
        best_stable_method = choice(list(methods.keys()))

    k = int(methods[best_stable_method]["results"][-1][0])

    # Salvar histórico de métricas em um arquivo .csv
    metrics_df = pd.DataFrame()
    for method_name in metrics_history.keys():
        method_df = pd.DataFrame(metrics_history[method_name])
        method_df['method'] = method_name
        metrics_df = pd.concat([metrics_df, method_df])

    metrics_df.to_csv(parsed_args.metrics_output_file, index=False)

    # Gerar gráficos para cada método
    for method_name in methods.keys():
        plt.figure()
        plt.plot(metrics_history[method_name]['features'], metrics_history[method_name]['accuracy'], label='Acurácia')
        plt.plot(metrics_history[method_name]['features'], metrics_history[method_name]['precision'], label='Precisão')
        plt.plot(metrics_history[method_name]['features'], metrics_history[method_name]['recall'], label='Recall')
        plt.plot(metrics_history[method_name]['features'], metrics_history[method_name]['f1'], label='F1')

        plt.xlabel('Número de features')
        plt.ylabel('Métricas')
        plt.title(f'Desempenho de {method_name} ao longo do processo de seleção')
        plt.legend()
        plt.show()

    return best_stable_method, k

if __name__=="__main__":
    start = time.time()
    parsed_args = parse_args(sys.argv[1:])
    X, y = get_X_y(parsed_args, get_dataset(parsed_args))
    total_features = get_dataset(parsed_args).shape[1] - 1
    print(f'Number of Features >> {X.shape[1]}')

    if(parsed_args.initial_n_features > total_features):
        print(f"ERRO: --initial-n-features ({parsed_args.initial_n_features}) maior que a qtd de features do dataset ({total_features})")
        exit(1)

    best_stable_method, lower_bound = selection_phase(X, y, methods, num_features=parsed_args.initial_n_features, increment=parsed_args.increment)
    print(f'Menor limite inferior encontrado: {best_stable_method}, {lower_bound}')

    new_X = correlation_phase(X, y, lower_bound, best_stable_method, methods)
    new_X.to_csv(parsed_args.output_file, index=False)
    print("Dataset final criado")

    stop = time.time()
    total_time = stop - start
    print(f'Time: {total_time: .2f} segundos.')