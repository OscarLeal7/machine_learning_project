# Importe as bibliotecas necessárias
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Defina uma função para avaliar o modelo
def evaluate_model(y_true, y_pred):
    # Calcule as métricas de avaliação
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # Imprima as métricas
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    
    # Retorne as métricas para possível uso posterior
    return accuracy, precision, recall, f1

# Se este arquivo for executado como um script, execute a função de avaliação com dados de exemplo
if __name__ == "__main__":
    # Exemplo de rótulos verdadeiros e predições
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    
    # Avalie o modelo
    evaluate_model(y_true, y_pred)
