from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import os

app = Flask(__name__, template_folder=".")

# Carregar os dados e treinar o modelo
file_path = "wine_data.csv"
data = pd.read_csv(file_path)

X = data.drop(columns=['quality'])
y = data['quality']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_scaled, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        dados = request.get_json()
        novo_vinho_df = pd.DataFrame([dados], columns=X.columns)
        novo_vinho_normalizado = scaler.transform(novo_vinho_df)
        qualidade_predita = rf.predict(novo_vinho_normalizado)[0]
        probabilidades = rf.predict_proba(novo_vinho_normalizado).tolist()

        return jsonify({
            "qualidade_predita": int(qualidade_predita),
            "probabilidades": probabilidades
        })

    except Exception as e:
        return jsonify({"erro": str(e)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

