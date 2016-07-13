"""
@author: Nan-Tsou Liu
created_at: 2016-07-10

Api for provide the pipline and parameter for fraud person-of-interest (POI) prediction model.


"""
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans


def get_LogReg_pipeline():

    pipeline = Pipeline(steps=[('minmaxer', MinMaxScaler()),
                               ('selection', SelectKBest(score_func=f_classif)),
                               ('reducer', PCA()),
                               ('classifier', LogisticRegression())])
    return pipeline


def get_LogReg_params():

    params = {'reducer__n_components': [0.5],
              'reducer__whiten': [True],
              'selection__k': [17],
              'classifier__C': [0.5],
              'classifier__class_weight': ['auto'],
              'classifier__tol': [1e-64]}
    return params

def get_LSVC_pipeline():

    pipeline = Pipeline(steps=[('minmaxer', MinMaxScaler()),
                               ('selection', SelectKBest(score_func=f_classif)),
                               ('reducer', PCA()),
                               ('classifier', LinearSVC())])
    return pipeline


def get_LSVC_params():

    params = {'reducer__n_components': [0.5],
              'reducer__whiten': [True],
              'selection__k': [15],
              'classifier__C': [0.15],
              'classifier__tol': [1e-32],
              'classifier__class_weight': ['auto'],
              'classifier__random_state': [42]}
    return params


def get_SVC_pipeline():

    pipeline = Pipeline(steps=[('minmaxer', MinMaxScaler()),
                               ('selection', SelectKBest(score_func=f_classif)),
                               ('reducer', PCA()),
                               ('classifier', SVC())])
    return pipeline


def get_SVC_params():

    params = {'reducer__n_components': [0.5],
              'reducer__whiten': [False],
              'selection__k': [15],
              'classifier__C': [0.01],
              'classifier__gamma': [0.0],
              'classifier__kernel': ['rbf'],
              'classifier__tol': [1e-8],
              'classifier__class_weight': ['auto'],
              'classifier__random_state': [42]}
    return params

def get_DTree_pipeline():
    pipeline = Pipeline(steps=[('minmaxer', MinMaxScaler()),
     ('selection', SelectKBest(score_func=f_classif)),
     ('reducer', PCA()),
     ('classifier', DecisionTreeClassifier())])
    return pipeline


def get_DTree_params():
    params = {'reducer__n_components': [0.5],
              'reducer__whiten': [False, True],
              'selection__k': [13, 15, 17],
              'classifier__criterion': ['gini', 'entropy'],
              'classifier__max_depth': [1, 2, 4, 8],
              'classifier__min_samples_leaf': [1, 2, 4, 8],
              'classifier__min_samples_split': [1, 2, 4, 8],
              'classifier__class_weight': ['auto', 'None']}
    return params



def get_KMeans_pipeline():
    pipeline = Pipeline(steps=[('minmaxer', MinMaxScaler()),
     ('selection', SelectKBest(score_func=f_classif)),
     ('reducer', PCA()),
     ('classifier', KMeans())])
    return pipeline


def get_KMeans_params():
    params = {'reducer__n_components': [4],
              'reducer__whiten': [False],
              'selection__k': [10],
              'classifier__n_clusters': [2],
              'classifier__n_init': [100],
              'classifier__init': ['k-means++'],
              'classifier__tol': [1e-8],
              'classifier__random_state': [42]}
    return params
