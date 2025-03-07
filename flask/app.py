from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Inicializar Flask
app = Flask(__name__)

# 1. Carregar os dados e treinar o modelo (simulação)
file_path = "wine_data.csv"
data = pd.read_csv(file_path)

X = data.drop(columns=['quality'])
y = data['quality']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_scaled, y)

# 2. Criar a rota para previsão
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receber os dados do vinho no formato JSON
        dados = request.get_json()

        # Transformar em DataFrame com as colunas corretas
        novo_vinho_df = pd.DataFrame([dados], columns=X.columns)

        # Normalizar os dados
        novo_vinho_normalizado = scaler.transform(novo_vinho_df)

        # Fazer a previsão
        qualidade_predita = rf.predict(novo_vinho_normalizado)[0]
        probabilidades = rf.predict_proba(novo_vinho_normalizado).tolist()

        # Retornar resposta JSON
        return jsonify({
            "qualidade_predita": int(qualidade_predita),
            "probabilidades": probabilidades
        })

    except Exception as e:
        return jsonify({"erro": str(e)})

# 3. Rodar o servidor Flask
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

