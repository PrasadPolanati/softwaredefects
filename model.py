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
        print(train_data.dtypes)
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
    

    def modelrf(self):
        X_train,Y_train = self.training_data(index_col='id')
        RF_md = RandomForestClassifier(n_estimators = 500, 
                                   max_depth = 7,
                                   min_samples_split = 15,
                                   min_samples_leaf = 10).fit(X_train, Y_train)
    
        # RF_pred = RF_md.predict_proba(X_test)[:, 1]
        # RF_score = roc_auc_score(Y_test, RF_pred)

        # print('Fold', i, '==> RF oof ROC-AUC score is ==>', RF_score)

        # RF_pred_test = RF_md.predict_proba(test_cv)[:, 1]

        #save the model using joblib
        joblib.dump(RF_md, r'C:\Users\prasa\Desktop\softwaredefects\data\randomforest_model.joblib')
        return RF_md


    def modelextratrees(self):
        X_train,Y_train = self.training_data(index_col='id')
        ET_md = ExtraTreesClassifier(n_estimators = 500, 
                                    max_depth = 7,
                                    min_samples_split = 15,
                                    min_samples_leaf = 10).fit(X_train, Y_train)

        # ET_pred = ET_md.predict_proba(X_test)[:, 1]
        # ET_score = roc_auc_score(Y_test, ET_pred)

        # print('Fold', i, '==> ET oof ROC-AUC score is ==>', ET_score)

        # ET_pred_test = ET_md.predict_proba(test_cv)[:, 1]
        #save the model using joblib
        joblib.dump(ET_md, r'C:\Users\prasa\Desktop\softwaredefects\data\extractrees_model.joblib')
        return ET_md
    
    def modellightgbm(self):
        X_train,Y_train = self.training_data(index_col='id')
        LGBM_md = LGBMClassifier(objective = 'binary',
                             n_estimators = 500,
                             max_depth = 7,
                             learning_rate = 0.01,
                             num_leaves = 20,
                             reg_alpha = 3,
                             reg_lambda = 3,
                             subsample = 0.7,
                             colsample_bytree = 0.7).fit(X_train, Y_train)

        # lgb_pred = LGBM_md.predict_proba(X_test)[:, 1]
        # lgb_score = roc_auc_score(Y_test, lgb_pred)
        #save the model using joblib
        joblib.dump(LGBM_md, r'C:\Users\prasa\Desktop\softwaredefects\data\lightgbm_model.joblib')
        return LGBM_md

    def XGBoost(self):
        X_train,Y_train = self.training_data(index_col='id')
        XGB_md = XGBClassifier(objective = 'binary:logistic',
                            tree_method = 'hist',
                            colsample_bytree = 0.7, 
                            gamma = 2, 
                            learning_rate = 0.01, 
                            max_depth = 7, 
                            min_child_weight = 10, 
                            n_estimators = 500, 
                            subsample = 0.7).fit(X_train, Y_train)

        # xgb_pred = XGB_md.predict_proba(X_test)[:, 1]
        # xgb_score = roc_auc_score(Y_test, xgb_pred)
        #save the model using joblib
        joblib.dump(XGB_md, r'C:\Users\prasa\Desktop\softwaredefects\data\histxgbosst_model.joblib')
        return XGB_md


if __name__ == "__main__":
    # model_object  =  ModelTraining(data_path=r'C:\Users\prasa\Desktop\softwaredefects\data\train.csv').model()
    # model = joblib.load(r'C:\Users\prasa\Desktop\softwaredefects\data\xgboost_model.joblib')  # Replace with the correct path
    # model_object  =  ModelTraining(data_path=r'C:\Users\prasa\Desktop\softwaredefects\data\train.csv').modelrf()
    # print(model_object) 
    # model_object2 =  ModelTraining(data_path=r'C:\Users\prasa\Desktop\softwaredefects\data\train.csv').modelextratrees()
    # print(model_object2)
    # model_object3 =  ModelTraining(data_path=r'C:\Users\prasa\Desktop\softwaredefects\data\train.csv').modellightgbm()
    # print(model_object3)
    model_object4 =  ModelTraining(data_path=r'C:\Users\prasa\Desktop\softwaredefects\data\train.csv').XGBoost()
    print(model_object4)
 
    uploaded_file = r'C:\Users\prasa\Desktop\softwaredefects\data\test.csv'
    df = pd.read_csv(uploaded_file)
    df =  df.drop(columns=['id'])
    print(df.dtypes)
    predictions = model_object4.predict(df)
    print(predictions)
    # print(type(predictions))
    # csv_data = predictions.to_csv(index=False, header=False).encode('utf-8')
    
