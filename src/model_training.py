# Importar bibliotecas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras

# Gerar dados fictícios para demonstração
# Supondo que estamos trabalhando com um problema de regressão simples
np.random.seed(42)
X = np.random.rand(100, 5)  # 100 amostras, 5 características
y = np.random.rand(100)     # 100 amostras de saída

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalização dos dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Definir e treinar o modelo
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(X_train_scaled, y_train, epochs=100, validation_split=0.2)

# Avaliar o modelo
mse = model.evaluate(X_test_scaled, y_test)
print(f'Mean Squared Error: {mse}')

# Fazer previsões
y_pred = model.predict(X_test_scaled)

# Outras etapas de avaliação e visualização
plt.scatter(y_test, y_pred)
plt.xlabel('Valores Reais')
plt.ylabel('Valores Previstos')
plt.title('Valores Reais vs. Valores Previstos')
plt.show()
