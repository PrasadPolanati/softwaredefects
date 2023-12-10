import streamlit as st
import pandas as pd
import numpy as np
import matplotlib  as plt
import seaborn
import xgboost
import catboost
import lightgbm
import joblib
from io import BytesIO


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


# Load your XGBoost model
# model = joblib.load('xgboost_model.joblib')  # Replace with the correct path

st.title('Software Defects Prediction')


model_selection = {
    'Histogram Gradient Bossting Classifier ': joblib.load(r'xgboost_model.joblib'),
    'Random Forest ': joblib.load(r'randomforest_model.joblib'),
    'Extreme Gradient Boosting ': joblib.load(r'histxgbosst_model.joblib'),
    'Light GBM':joblib.load(r'lightgbm_model.joblib'),
    'Extra Trees Classfier':joblib.load(r'extractrees_model.joblib')
}



# dmatrix = xgboost.DMatrix(data=df)  # Assuming your model accepts XGBoost DMatrix

# Model selection for prediction
st.subheader("Select Model - entire dataset is used for training here")
model_selector = st.selectbox("Select a Model", list(model_selection.keys()))
selected_model = model_selection[model_selector]
# File Upload Widget
uploaded_file = st.file_uploader("Choose a file", type=["csv", "txt"])


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df =  df.drop(columns=['id'])

    predictions = selected_model.predict(df)
    predictions_df = pd.DataFrame(predictions, columns=['Predictions'])
    csv_data = predictions_df.to_csv(index=False, header=False).encode('utf-8')
    st.download_button(
        label="Download Predictions",
        data=csv_data,
        file_name="predictions.csv",
        key="download_button"
    )
else:
    print("No file is uploaded")

