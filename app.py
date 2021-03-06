import streamlit as st
import requests
from PIL import Image
import pickle
import pandas as pd
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
    return explainer, shap_values

@st.cache(hash_funcs={matplotlib.figure.Figure: hash})
def feature_importance_global_graphics(feature_inputs, shap_values):
    fig1 = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, feature_inputs, plot_type='bar')
    plt.show()
    fig2 = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values[0], feature_inputs)
    plt.show()
    return fig1, fig2

@st.cache(hash_funcs={matplotlib.figure.Figure: hash})
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

@st.cache(hash_funcs={matplotlib.figure.Figure: hash})
def boxplot_by_feature(df, feature_name):
    fig1 = plt.figure(figsize=(10, 8))
    sns.boxplot(data=df[[feature_name]], showfliers=False, showmeans=True)
    fig1.show()
    fig2 = plt.figure(figsize=(10, 8))
    sns.boxplot(x='predict', y=feature_name, showfliers=False, showmeans=True, data=df)
    fig2.show()
    return fig1, fig2

@st.cache(hash_funcs={matplotlib.figure.Figure: hash})
def barplot_by_feature(df, feature_name):
    fig = plt.figure(figsize=(10, 8))
    df = df.copy()
    df['Count'] = 1
    df = df[[feature_name, 'predict', 'Count']].groupby([feature_name, 'predict']).count().reset_index()
    sns.barplot(x=df[feature_name], y=df['Count'], hue=df['predict'])
    plt.xticks(rotation=0)
    fig.show()
    return fig

@st.cache
def feature_description(df_description, feature_name):
    return df_description[df_description.Row == feature_name].Description.values[0]

@st.cache
def get_client_predict(url):
    return requests.get(url=url)


# Data to predict
inputs = pd.read_csv('data/CustomerDataToBePredicted.csv')
inputs.sort_values(by='SK_ID_CURR', inplace=True)
inputs.set_index(keys='SK_ID_CURR', inplace=True)

# Preprocessor and model
preprocessor = pickle.load(open("models/preprocessor.pkl", "rb"))
lgbm_model = pickle.load(open("models/lgbm_model.pkl", "rb"))

# Shap explainer, values and mean values
lgbm_explainer, lgbm_shap_values = calculate_explainer_shapval_and_means(lgbm_model, inputs)

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

list_of_some_ids = list_of_worst_customer_ids + list_of_best_customer_ids
list_of_some_ids.sort()

# Data feature description
hc_col_desc = pd.read_csv('data/home_credit_feature_description.csv')

# API pr??diction
api_url = 'https://api-flask-projet-home-credit.herokuapp.com/'
str_predict = 'predict/'


col1, col2, col3 = st.columns(3)
with col1:
    st.write(" ")
with col2:
    image = Image.open('img/logo.png')
    st.image(image, caption='Pr??t ?? d??penser')
with col3:
    st.write(" ")


st.write("""
# Application pour pr??dire l'accord ou non d'un pr??t
""")

st.subheader('Data')
st.write(inputs.loc[list_of_some_ids])

if st.button("Description LightGBM Classifier"):
    st.write("Le mod??le de pr??diction utilis?? ici est le LightGBM Classifier, l'un des algorithmes ML les plus performants. \
             LightGBM est un cadre de boosting de gradient rapide, distribu?? et haute performance bas?? sur des algorithmes d'arbre \
             de d??cision, utilis?? pour le classement, la classification et de nombreuses autres t??ches d'apprentissage automatique.")
else:
    st.write("Mod??le LightGBM Classifier")

if st.button("Description M??thode Shap"):
    st.write("### But:")
    st.write(
    "Calculer la valeur de Shapley pour toutes les variables ?? chaque client du dataset. \
    Cette approche explique la sortie du mod??le par la somme des effets de chaque variable. \
    Ils se basent sur la valeur de Shapley qui provient de la th??orie des jeux. \
    L???id??e est de moyenner l???impact qu???une variable a pour toutes les combinaisons de variables possibles.")
    st.write("### Locale:")
    st.write(
    "Gr??ce ?? la valeur de Shap, on peut d??terminer l???effet des diff??rentes variables d???une pr??diction pour un mod??le \
    qui explique l?????cart de cette pr??diction par rapport ?? la valeur de base.")
    st.write("### Globale:")
    st.write("En moyennant les valeurs absolues des valeurs de Shap pour chaque variable, \
    nous pouvons remonter ?? l???importance globale des variables.")
else:
    st.write("M??thode shap (SHapley Additive exPlanation)")

st.subheader('Approche globale')
st.write("#### Visualisation de l???importance des variables du mod??le de mani??re globale")
# Print feature importance global with shap
graph_mean_global = feature_importance_global_graphics(inputs, lgbm_shap_values)[0]
graph_shap_global = feature_importance_global_graphics(inputs, lgbm_shap_values)[1]
st.write(graph_mean_global)
st.write("- Sur cette image, l???importance des variables est calcul??e en moyennant la valeur absolue des valeurs de Shap.")
st.write(graph_shap_global)
st.write("- Sur cette image, les valeurs de Shap sont repr??sent??es pour chaque variable dans leur ordre d???importance, \
chaque point repr??sente une valeur de Shap (pour un exemple), les points rouges repr??sentent des valeurs ??lev??es de la variable \
et les points bleus des valeurs basses de la variable")

st.write("- Exemples d'ID de mauvais clients: ", list_of_worst_customer_ids)
st.write("- Exemples d'ID de bons clients: ", list_of_best_customer_ids)


option_id = st.selectbox('Quel ID voulez-vous voir ?', list_of_ids_in_order)
# list_of_some_ids
st.write('Vous avez s??lectionn??: ', option_id)

st.subheader('Pr??diction')
option_url = api_url + str_predict + str(option_id)
r = get_client_predict(option_url)
st.write(r.text)

st.subheader('Approche locale')
option_top_n_features = st.slider('Combien de crit??res voulez-vous voir ?', 10, inputs.shape[1])
st.write(feature_importance_local_graphics(id_=option_id, top_n_features=option_top_n_features))
st.write("- En rouge, les variables qui ont un impact positif \
(contribuent a?? ce que la pre??diction soit plus e??leve??e que la valeur de base) \
et, en bleu, celles ayant un impact ne??gatif \
(contribuent a?? ce que la pre??diction soit plus basse que la valeur de base)")


st.subheader('Graphiques')
st.write("- Selon si la variable choisie est discr??te (peut prendre toutes les valeurs possibles d'un intervalle de nombres) \
         ou continue (peut prendre uniquement certaines valeurs d'un intervalle de nombres) \
         le graphique sera respectivement un Barplot(Graphe en barre repr??sentant pour toutes les donn??es la variable ??tudi??e, \
         avec r??partition des pr??dictions accept??es/refus??es), ou un boxplot(Repr??sentation sch??matique des quartiles, m??diane et moyenne \
         pour toutes les donn??es de la variable ??tudi??e, avec r??partition des pr??dictions accept??es/refus??es)")
option_feature = st.selectbox('Quel crit??re particulier voulez-vous visualiser ?', all_variables)
st.write('Valeur: ', inputs.loc[option_id][option_feature])

if option_feature in continuous_variables:
    st.write("Repr??sentation sch??matique des quartiles, m??diane et moyenne")
    st.write(boxplot_by_feature(df=data_predict, feature_name=option_feature)[0])
    st.write("S??paration entre Pr??t accord?? et Pr??t refus??")
    st.write(boxplot_by_feature(df=data_predict, feature_name=option_feature)[1])
else:
    st.write("R??partition des valeurs et S??paration entre Pr??t accord?? et Pr??t refus??")
    st.write(barplot_by_feature(df=data_predict, feature_name=option_feature))

option_feature_description = st.selectbox('De quel crit??re voulez_vous voir la description ?', list(hc_col_desc.Row))
st.write(feature_description(df_description=hc_col_desc, feature_name=option_feature_description))


st.write("## Nouveaux crit??res")
st.write("### Par le calcul")
st.write("- DAYS_EMPLOYED_PERC = DAYS_EMPLOYED / DAYS_BIRTH")
st.write("- INCOME_CREDIT_PERC = AMT_INCOME_TOTAL / AMT_CREDIT")
st.write("- INCOME_PER_PERSON = AMT_INCOME_TOTAL / CNT_FAM_MEMBERS")
st.write("- ANNUITY_INCOME_PERC = AMT_ANNUITY/ AMT_INCOME_TOTAL")
st.write("- PAYMENT_RATE = AMT_ANNUITY / AMT_CREDIT")
st.write("- PAYMENT_PERC = AMT_PAYMENT / AMT_INSTALMENT (Percentage paid in each installment (amount paid and installment value))")
st.write("- PAYMENT_DIFF = AMT_INSTALMENT - AMT_PAYMENT (Difference paid in each installment (amount paid and installment value))")
st.write("- DPD = DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT (Days past due)")
st.write("- DBD = DAYS_INSTALMENT - DAYS_ENTRY_PAYMENT (days before due)")
st.write("### Par encodage(one hot encoder, pour les variables cat??gorielles)")
st.write("- Exemples: occupation type -> Tous les types d'occupation deviennent des crit??res binaire (non 0, oui 1), organization type etc..")
st.write("### Minimum, Maximum, Moyenne, Variance, Somme (Ajout suffixe)")
st.write("- Ajout de min, max, mean, var, sum en suffixe pour certaines variables")
st.write("### Ajout pr??fixe selon les donn??es d'o?? est extraite la variable")
st.write("- BURO_")
st.write("- ACTIVE_ (Cr??dit activ??)")
st.write("- CLOSED_ (Cr??dit ferm??)")
st.write("- POS_ ( (point of sales) and cash loans) et POS_COUNT (cash)")
st.write("- INSTAL_ et INSTAL_COUNT")
st.write("- CC_ et CC_COUNT (carte de cr??dit)")
