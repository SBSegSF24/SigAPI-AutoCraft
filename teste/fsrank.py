import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel, mutual_info_classif, RFE, SelectKBest, chi2

def calculateMutualInformationGain(features, target, k):
    """
    Calculate feature importance using Mutual Information Gain.

    Parameters:
    features (pd.DataFrame): Input features.
    target (pd.Series): Target variable.
    k (int): Number of top features to select.

    Returns:
    pd.DataFrame: DataFrame with top k features and their scores.
    """
    feature_names = features.columns
    mutualInformationGain = mutual_info_classif(features, target, random_state = 0)
    data = {'features': feature_names, 'score': mutualInformationGain}
    df = pd.DataFrame(data)
    df = df.sort_values(by = ['score'], ascending = False)
    return df[:k]
def calculateRandomForestClassifier(features, target, k):
    """
    Select top k features using Random Forest Classifier feature importance.

    Parameters:
    features (pd.DataFrame): Input features.
    target (pd.Series): Target variable.
    k (int): Number of top features to select.

    Returns:
    pd.DataFrame: DataFrame with top k features and their scores.
    """
    feature_names = np.array(features.columns.values.tolist())
    test = RandomForestClassifier(random_state = 0)
    test = test.fit(features, target)
    model = SelectFromModel(test, max_features = k, prefit = True)
    model.get_support()
    best_features = feature_names[model.get_support()]
    best_score = test.feature_importances_[model.get_support()]
    df = pd.DataFrame(list(zip(best_features, best_score)), columns = ['features', 'score']).sort_values(by = ['score'], ascending = False)
    return df

def calculateExtraTreesClassifier(features, target, k):
    """
    Select top k features using Extra Trees Classifier feature importance.

    Parameters:
    features (pd.DataFrame): Input features.
    target (pd.Series): Target variable.
    k (int): Number of top features to select.

    Returns:
    pd.DataFrame: DataFrame with top k features and their scores.
    """
    feature_names = np.array(features.columns.values.tolist())
    test = ExtraTreesClassifier(random_state = 0)
    test = test.fit(features, target)
    model = SelectFromModel(test, max_features = k, prefit = True)
    model.get_support()
    best_features = feature_names[model.get_support()]
    best_score = test.feature_importances_[model.get_support()]
    df = pd.DataFrame(list(zip(best_features,best_score)), columns = ['features','score']).sort_values(by = ['score'], ascending = False)
    return df

def calculateRFERandomForestClassifier(features, target, k):
    """
    Select top k features using RFE with Random Forest Classifier.

    Parameters:
    features (pd.DataFrame): Input features.
    target (pd.Series): Target variable.
    k (int): Number of top features to select.

    Returns:
    pd.DataFrame: DataFrame with top k features and their scores.
    """
    feature_names = np.array(features.columns.values.tolist())
    rfe = RFE(estimator = RandomForestClassifier(random_state = 0), n_features_to_select = k)
    model = rfe.fit(features, target)
    best_features = feature_names[model.support_]
    best_scores = rfe.estimator_.feature_importances_
    df = pd.DataFrame(list(zip(best_features, best_scores)), columns = ['features', 'score']).sort_values(by = ['score'], ascending = False)
    return df

def calculateRFEGradientBoostingClassifier(features, target, k):
    """
    Select top k features using RFE with Gradient Boosting Classifier.

    Parameters:
    features (pd.DataFrame): Input features.
    target (pd.Series): Target variable.
    k (int): Number of top features to select.

    Returns:
    pd.DataFrame: DataFrame with top k features and their scores.
    """
    feature_names= np.array(features.columns.values.tolist())
    rfe = RFE(estimator = GradientBoostingClassifier(random_state = 0), n_features_to_select = k)
    model = rfe.fit(features, target)
    best_features = feature_names[model.support_]
    best_scores = rfe.estimator_.feature_importances_
    df = pd.DataFrame(list(zip(best_features, best_scores)), columns = ['features', 'score']).sort_values(by = ['score'], ascending=False)
    return df

def calculateSelectKBest(features, target, k):
    """
    Select top k features using SelectKBest with chi-squared statistic.

    Parameters:
    features (pd.DataFrame): Input features.
    target (pd.Series): Target variable.
    k (int): Number of top features to select.

    Returns:
    pd.DataFrame: DataFrame with top k features and their scores.
    """
    feature_names= np.array(features.columns.values.tolist())
    chi2_selector= SelectKBest(score_func = chi2, k= k)
    chi2_selector.fit(features, target)
    chi2_scores = pd.DataFrame(list(zip(feature_names,chi2_selector.scores_)), columns = ['features','score'])
    df = pd.DataFrame(list(zip(feature_names,chi2_selector.scores_)), columns = ['features','score']).sort_values(by = ['score'], ascending = False)
    return df[:k]

methods = {
    'mutualInformation': {'function': calculateMutualInformationGain, 'results': [], 'is_stable': False},
    'selectRandom': {'function': calculateRandomForestClassifier, 'results': [], 'is_stable': False},
    'selectExtra': {'function': calculateExtraTreesClassifier, 'results': [], 'is_stable': False},
    #'RFERandom': {'function': calculateRFERandomForestClassifier, 'results': [], 'is_stable': False},
    #'RFEGradient': {'function': calculateRFEGradientBoostingClassifier, 'results': [], 'is_stable': False},
    'selectKBest': {'function': calculateSelectKBest, 'results': [], 'is_stable': False}
}
