import pickle
import pandas as pd
import flask
from flask import Flask, jsonify, request, render_template


# Load preprocessor and lgbm model
preprocessor = pickle.load(open("models/preprocessor.pkl", "rb"))
lgbm_model = pickle.load(open("models/lgbm_model.pkl", "rb"))

# Load sample of data (X test) where client ids are indexes
inputs = pd.read_csv('data/CustomerDataToBePredicted.csv')
inputs.sort_values(by='SK_ID_CURR', inplace=True)
inputs.set_index(keys='SK_ID_CURR', inplace=True)


def predict_class_and_proba_customer(data, id_, preprocess, model):
    data_id_customer = data.loc[id_].to_frame().T
    data_customer_preprocess = preprocess.transform(data_id_customer)
    y_pred = model.predict(data_customer_preprocess)[0]
    y_prob = model.predict_proba(data_customer_preprocess)[0].tolist()
    if y_pred == 0:
        return "Prêt accordé" + " ->  Probabilité:" + f"{y_prob[0]}"
    else:
        return "Prêt Refusé" + " ->  Probabilité:" + f"{y_prob[1]}"


app = Flask(__name__)

#url = 'https://flask-api.heroku-app.com/results'

@app.route('/')
def home():
    return 'Prédiction de prêt'


@app.route('/predict/<customer_id>', methods=['GET', 'POST'])
def predict(customer_id):
    results = predict_class_and_proba_customer(data=inputs, id_=int(customer_id), preprocess=preprocessor, model=lgbm_model)
    return results


if __name__ == "__main__":
    app.run(debug=True)

