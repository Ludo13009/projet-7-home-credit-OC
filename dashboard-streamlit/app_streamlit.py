import streamlit as st
import pickle
import pandas as pd
import numpy as np
import shap
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


@st.cache
def calculate_explainer_shapval_and_means(classifier, feature_inputs):
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(feature_inputs)
    mean_values = np.abs(shap_values).mean(0)
    return explainer, shap_values, mean_values

@st.cache(hash_funcs={matplotlib.figure.Figure: hash})
def feature_importance_global_graphics(feature_inputs, shap_values):
    fig1 = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, feature_inputs, plot_type='bar')
    plt.show()
    fig2 = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values[0], feature_inputs)
    plt.show()
    return fig1, fig2


def predict_class_and_proba_customer(data, id_, preprocess, model):
    data_id_customer = data.loc[id_].to_frame().T
    data_customer_preprocess = preprocess.transform(data_id_customer)
    y_pred = model.predict(data_customer_preprocess)[0]
    y_prob = model.predict_proba(data_customer_preprocess)[0].tolist()
    if y_pred == 0:
        return "Prêt accordé" + " ->  Probabilité:" + f"{y_prob[0]}"
    else:
        return "Prêt Refusé" + " ->  Probabilité:" + f"{y_prob[1]}"


def feature_importance_local_graphics(id_, top_n_features):
    index_id_ = list_of_ids_in_order.index(id_)
    fig = plt.figure(figsize=(10, 8))
    shap.plots._waterfall.waterfall_legacy(lgbm_explainer.expected_value[1], lgbm_shap_values[0][index_id_],
                                           features=inputs.loc[id_], max_display=top_n_features)
    plt.show()
    return fig

@st.cache
def predict_all_data(data, preprocess, model):
    data_copy = data.copy()
    data_customer_preprocess = preprocess.transform(data_copy)
    y_pred = model.predict(data_customer_preprocess).tolist()
    y_prob = model.predict_proba(data_customer_preprocess).tolist()
    data_copy['predict'] = y_pred
    data_copy['list_of_proba'] = y_prob
    return data_copy

def extract_proba(x, i):
    return x[i]

def boxplot_by_feature(df, feature_name):
    fig1 = plt.figure(figsize=(10, 8))
    sns.boxplot(data=df[[feature_name]], showfliers=False, showmeans=True)
    fig1.show()
    fig2 = plt.figure(figsize=(10, 8))
    sns.boxplot(x='predict', y=feature_name, showfliers=False, showmeans=True, data=df)
    fig2.show()
    return fig1, fig2

def barplot_by_feature(df, feature_name):
    fig = plt.figure(figsize=(10, 8))
    df = df.copy()
    df['iniatialize'] = 1
    df = df[[feature_name, 'predict', 'iniatialize']].groupby([feature_name, 'predict']).count().reset_index()
    sns.barplot(x=df[feature_name], y=df['iniatialize'], hue=df['predict'])
    plt.xticks(rotation=0)
    fig.show()
    return fig


def feature_description(df_description, feature_name):
    return df_description[df_description.Row == feature_name].Description.values[0]



## Load all data

# Data to predict
inputs = pd.read_csv('data/CustomerDataToBePredicted.csv')
inputs.sort_values(by='SK_ID_CURR', inplace=True)
inputs.set_index(keys='SK_ID_CURR', inplace=True)

# Preprocessor and model
preprocessor = pickle.load(open("models/preprocessor.pkl", "rb"))
lgbm_model = pickle.load(open("models/lgbm_model.pkl", "rb"))

# Shap explainer, values and mean values
lgbm_explainer, lgbm_shap_values, lgbm_mean_values = calculate_explainer_shapval_and_means(lgbm_model, inputs)

# Discrete, continuous and all variables
discrete_variables = sorted(inputs.loc[:, inputs.nunique() < 10].columns.tolist())
continuous_variables = sorted(inputs.loc[:, inputs.nunique() >= 10].columns.tolist())
all_variables = sorted(continuous_variables + discrete_variables)

# Data predict
data_predict = predict_all_data(inputs, preprocessor, lgbm_model)

# List of ids, worst ids, best ids
list_of_ids_in_order = inputs.index.tolist()

data_predict_1 = data_predict[data_predict.predict == 1]
data_predict_1['proba'] = data_predict_1.list_of_proba.apply(extract_proba, i=1)
list_of_worst_customer_ids = data_predict_1[data_predict_1.proba > 0.95].head(10).index.tolist()

data_predict_0 = data_predict[data_predict.predict == 0]
data_predict_0['proba'] = data_predict_0.list_of_proba.apply(extract_proba, i=0)
list_of_best_customer_ids = data_predict_0[data_predict_0.proba > 0.95].head(10).index.tolist()

# Data feature description
hc_col_desc = pd.read_csv('data/home_credit_feature_description.csv')



st.title(" Application 'Prêt à dépenser' ")

st.write("""
## 1. Application pour prédire l'accord d'un prêt
### 2. Visualisation global et local des critères principaux ayant abouti à cette prédiction
""")

st.subheader('Data')
st.write(inputs.head())

st.subheader('Feature importance global')
# Print feature importance global with shap
graph_mean_global = feature_importance_global_graphics(inputs, lgbm_shap_values)[0]
graph_shap_global = feature_importance_global_graphics(inputs, lgbm_shap_values)[1]
st.write(graph_mean_global)
st.write(graph_shap_global)

st.write('Examples ids of bad clients: ', list_of_worst_customer_ids)
st.write('Examples ids of good clients:', list_of_best_customer_ids)


option_id = st.selectbox('What ids do you want ?', list_of_ids_in_order)
st.write('You selected:', option_id)

st.subheader('Prédiction')
st.write(predict_class_and_proba_customer(data=inputs, id_=option_id, preprocess=preprocessor, model=lgbm_model))


option_top_n_features = st.slider('How much top features do you see ?', 1, inputs.shape[1])
st.write(feature_importance_local_graphics(id_=option_id, top_n_features=option_top_n_features))


option_feature = st.selectbox('What feature do you see graph ?', all_variables)
st.write('Value: ', inputs.loc[option_id][option_feature])

if option_feature in continuous_variables:
    st.write(boxplot_by_feature(df=data_predict, feature_name=option_feature)[0])
    st.write(boxplot_by_feature(df=data_predict, feature_name=option_feature)[1])
else:
    st.write(barplot_by_feature(df=data_predict, feature_name=option_feature))


option_feature_description = st.selectbox('What feature do you see description ?', list(hc_col_desc.Row))
st.write(feature_description(df_description=hc_col_desc, feature_name=option_feature_description))


