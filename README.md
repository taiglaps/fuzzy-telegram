# fuzzy-telegram
Classificador de E-mail. 
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Simulando os dados reais e previstos (0 = Não é Spam, 1 = Spam)
np.random.seed(42)  # Para resultados reprodutíveis
y_real = np.random.randint(0, 2, size=100)  # Valores reais
y_pred = np.random.randint(0, 2, size=100)  # Valores previstos

# Calculando a matriz de confusão
cm = confusion_matrix(y_real, y_pred)

# Calculando as métricas
accuracy = accuracy_score(y_real, y_pred)
precision = precision_score(y_real, y_pred)
recall = recall_score(y_real, y_pred)
f1 = f1_score(y_real, y_pred)

# Exibindo os resultados
print("Matriz de Confusão:\n", cm)
print(f"Acurácia: {accuracy:.2f}")
print(f"Precisão: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
