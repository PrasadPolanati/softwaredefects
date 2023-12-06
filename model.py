##importing the basic libraraies
import pandas as pd
import numpy as np
import streamlit
import joblib
from joblib import memory


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


class ModelTraining(object):

    def __init__(self,data_path):
        self.data_path = data_path


    def training_data(self,index_col):
        train_data = pd.read_csv(self.data_path,index_col= index_col)
        X = train_data.drop(['defects'],axis=1)
        y= train_data['defects']
        return X,y
    

    def model(self):
        X_train,Y_train = self.training_data(index_col='id')

        hist_md = HistGradientBoostingClassifier(l2_regularization = 0.01,
                                            early_stopping = False,
                                            learning_rate = 0.01,
                                            max_iter = 500,
                                            max_depth = 5,
                                            max_bins = 255,
                                            min_samples_leaf = 15,
                                            max_leaf_nodes = 10).fit(X_train, Y_train)
        

        ##save the model using joblib
        joblib.dump(hist_md, r'C:\Users\prasa\Desktop\softwaredefects\data\xgboost_model.joblib')

        # hist_pred = hist_md.predict_proba(X_test)[:, 1]
        # hist_score = roc_auc_score(Y_test, hist_pred)
        # print('Fold ==> Hist oof ROC-AUC score is ==>', hist_score)  
        return hist_md



if __name__ == "__main__":
    model_object  =  ModelTraining(data_path=r'C:\Users\prasa\Desktop\softwaredefects\data\train.csv').model()
    model = joblib.load(r'C:\Users\prasa\Desktop\softwaredefects\data\xgboost_model.joblib')  # Replace with the correct path
    uploaded_file = r'C:\Users\prasa\Desktop\softwaredefects\data\test.csv'
    df = pd.read_csv(uploaded_file)
    df =  df.drop(columns=['id'])
    predictions = model.predict(df)
    print(predictions)
    print(type(predictions))
    # csv_data = predictions.to_csv(index=False, header=False).encode('utf-8')
    

#r'C:\Users\prasa\Desktop\data\train.csv'
#index_col='id'