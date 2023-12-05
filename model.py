##importing the basic libraraies
import pandas as pd
import numpy as np
import streamlit
import joblib


from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import roc_auc_score, roc_curve, make_scorer, f1_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler, LabelEncoder


from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


class Enseblemodel(object):

    def __init__(self):
        pass


    def training_data(self,path,index_col):
        train_data = pd.read_csv(path,index_col= index_col)
        X = train_data.drop(['defetcs'],axis=1)
        y= train_data['defects']
        return X,y
    

    def hill_climbing(self):
        return None



    
    def model(self):
        model_object = None

        return model_object





#r'C:\Users\prasa\Desktop\data\train.csv'
#index_col='id'