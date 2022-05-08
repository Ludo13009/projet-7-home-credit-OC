import streamlit as st
from PIL import Image
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



col1, col2, col3 = st.columns(3)
with col1:
    st.write(" ")
with col2:
    image = Image.open('img/logo.png')
    st.image(image, caption='Prêt à dépenser')
with col3:
    st.write(" ")



st.write("""
# Application pour prédire l'accord ou non d'un prêt
""")

st.subheader('Data')
st.write(inputs.head())

if st.button("Description LightGBM Classifier"):
    st.write("Le modèle de prédiction utilisé ici est le LightGBM Classifier, l'un des algorithmes ML les plus performants. \
             LightGBM est un cadre de boosting de gradient rapide, distribué et haute performance basé sur des algorithmes d'arbre \
             de décision, utilisé pour le classement, la classification et de nombreuses autres tâches d'apprentissage automatique.")
else:
    st.write("Modèle LightGBM Classifier")

if st.button("Description Méthode Shap"):
    st.write("### But:")
    st.write(
    "Calculer la valeur de Shapley pour toutes les variables à chaque client du dataset. \
    Cette approche explique la sortie du modèle par la somme des effets de chaque variable. \
    Ils se basent sur la valeur de Shapley qui provient de la théorie des jeux. \
    L’idée est de moyenner l’impact qu’une variable a pour toutes les combinaisons de variables possibles.")
    st.write("### Locale:")
    st.write(
    "Grâce à la valeur de Shap, on peut déterminer l’effet des différentes variables d’une prédiction pour un modèle \
    qui explique l’écart de cette prédiction par rapport à la valeur de base.")
    st.write("### Globale:")
    st.write("En moyennant les valeurs absolues des valeurs de Shap pour chaque variable, \
    nous pouvons remonter à l’importance globale des variables.")
else:
    st.write("Méthode shap (SHapley Additive exPlanation)")

st.subheader('Approche globale')
st.write("#### Visualisation de l’importance des variables du modèle de manière globale")
# Print feature importance global with shap
graph_mean_global = feature_importance_global_graphics(inputs, lgbm_shap_values)[0]
graph_shap_global = feature_importance_global_graphics(inputs, lgbm_shap_values)[1]
st.write(graph_mean_global)
st.write("- Sur cette image, l’importance des variables est calculée en moyennant la valeur absolue des valeurs de Shap.")
st.write(graph_shap_global)
st.write("- Sur cette image, les valeurs de Shap sont représentées pour chaque variable dans leur ordre d’importance, \
chaque point représente une valeur de Shap (pour un exemple), les points rouges représentent des valeurs élevées de la variable \
et les points bleus des valeurs basses de la variable")

st.write("- Exemples d'ID de mauvais clients: ", list_of_worst_customer_ids)
st.write("- Exemples d'ID de bons clients: ", list_of_best_customer_ids)


option_id = st.selectbox('Quel ID voulez-vous voir ?', list_of_ids_in_order)
st.write('Vous avez sélectionné: ', option_id)

st.subheader('Prédiction')
st.write(predict_class_and_proba_customer(data=inputs, id_=option_id, preprocess=preprocessor, model=lgbm_model))

st.subheader('Approche locale')
option_top_n_features = st.slider('Combien de critères voulez-vous voir ?', 10, inputs.shape[1])
st.write(feature_importance_local_graphics(id_=option_id, top_n_features=option_top_n_features))
st.write("- En rouge, les variables qui ont un impact positif \
(contribuent à ce que la prédiction soit plus élevée que la valeur de base) \
et, en bleu, celles ayant un impact négatif \
(contribuent à ce que la prédiction soit plus basse que la valeur de base)")


st.subheader('Graphiques')
st.write("- Selon si la variable choisie est discrète (peut prendre toutes les valeurs possibles d'un intervalle de nombres) \
         ou continue (peut prendre uniquement certaines valeurs d'un intervalle de nombres) \
         le graphique sera respectivement un Barplot(Graphe en barre représentant pour toutes les données la variable étudiée, \
         avec répartition des prédictions acceptées/refusées), ou un boxplot(Représentation schématique des quartiles, médiane et moyenne \
         pour toutes les données de la variable étudiée, avec répartition des prédictions acceptées/refusées)")
option_feature = st.selectbox('Quel critère particulier voulez-vous visualiser ?', all_variables)
st.write('Valeur: ', inputs.loc[option_id][option_feature])

if option_feature in continuous_variables:
    st.write(boxplot_by_feature(df=data_predict, feature_name=option_feature)[0])
    st.write(boxplot_by_feature(df=data_predict, feature_name=option_feature)[1])
else:
    st.write(barplot_by_feature(df=data_predict, feature_name=option_feature))


option_feature_description = st.selectbox('De quel critère voulez_vous voir la description ?', list(hc_col_desc.Row))
st.write(feature_description(df_description=hc_col_desc, feature_name=option_feature_description))
