import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import os

processed_data_path = os.path.join('data', 'processed', 'processed_data.csv')

def load_processed_data(path):
    return pd.read_csv(path)

def train_model(X_train, y_train):
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(X_train, y_train, epochs=100, validation_split=0.2)
    return model

if __name__ == "__main__":
    # Carregar os dados processados
    processed_data = load_processed_data(processed_data_path)

    # Separar recursos (features) e alvo (target)
    X = processed_data.drop('target', axis=1)  # Certifique-se de que 'target' é o nome da coluna do alvo
    y = processed_data['target']

    # Dividir os dados em conjunto de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Escalar os dados
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Treinar o modelo
    model = train_model(X_train_scaled, y_train)

    # Avaliar o modelo
    mse = model.evaluate(X_test_scaled, y_test)
    print(f'Mean Squared Error: {mse}')

    # Fazer previsões
    y_pred = model.predict(X_test_scaled)

    # Plotar valores reais vs. valores previstos
    plt.scatter(y_test, y_pred)
    plt.xlabel('Valores Reais')
    plt.ylabel('Valores Previstos')
    plt.title('Valores Reais vs. Valores Previstos')
    plt.show()
