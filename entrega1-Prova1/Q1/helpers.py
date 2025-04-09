from sklearn.preprocessing import MinMaxScaler
import numpy as np

def get_scaler():
    # Definição dos valores min e max (de acordo com o dataset)
    feature_ranges = {
        "media_raio": (6.00, 26.00),
        "media_textura": (9.00, 34.00),
        "media_perimetro": (52.00, 174.00),
        "media_area": (157.00, 2051.00),          # Taxa de juros em percentual
        "mean_smoothness": (0.06, 0.13),
    }

    # Criar scalers para cada recurso
    scalers = {}
    for feature, (min_val, max_val) in feature_ranges.items():
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(np.array([min_val, max_val]).reshape(-1, 1))
        scalers[feature] = scaler
    return scalers