import os
import joblib
from flask import Flask, render_template, request   # jsonify
from helpers import get_scaler
import numpy as np
import pandas as pd

# Configurações do Flask
app = Flask(__name__)

# Configs para pegar caminho absoluto do modelo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Carregando o modelo
modelo_path = os.path.join(BASE_DIR, "modelo", "modelo_DT_cancer.pkl")
modelo = joblib.load(modelo_path)

# normalizer -> "helpers.py"
scalers = get_scaler()

# Index
@app.route("/")
def formulario():
    return render_template("index.html")

# Resultado
@app.route("/result", methods=["POST"])
def prever():
    try:
        dados = request.form
        client_name = dados.get("pacient_name")


        # Converte e organiza os dados
        entrada = {
            "media_raio": float(dados.get("media_raio")),
            "media_textura": float(dados.get("media_textura")),
            "media_perimetro": float(dados.get("media_perimetro")),
            "media_area": float(dados.get("media_area")),
            "mean_smoothness": float(dados.get("mean_smoothness")),
        }

        # Lista para guardar os dados normalizados
        dados_normalizados = []
        # Itera sobre as colunas numéricas para normalizar os valores
        for col in entrada:
            valor_original = entrada[col]                                       # valor do form
            valor_transformado = scalers[col].transform([[valor_original]])     # normalização do valor do form
            valor_normalizado = valor_transformado[0][0]                        # extrai doarray
            dados_normalizados.append(valor_normalizado)                        # Entra na lista

        # Adição dos categóricos
        entrada_modelo = dados_normalizados
        
        # Usa o modelo .pkl para realizar a predição
        previsao = modelo.predict([entrada_modelo])

        # Resultado da previsão (int)
        resultado = int(previsao[0])

        # Resultado
        if (resultado == 0):
            mensagem = "Tumor Benigno!"
            cor = "success"                         # cor verde (bootstrap)
        else:
            mensagem = "Tumor maligno"
            cor = "danger"                          # cor vermelha (bootstrap)
        return render_template("result.html", client_name=client_name, mensagem=mensagem, cor=cor)
    except Exception as e:
        return render_template("result.html", client_name="Erro. Tente novamente", mensagem=str(e), cor="danger")

if __name__ == "__main__":
    app.run(debug = True)