import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import numpy as np
import pickle
import time
import os

caminho_arquivo = "/home/matheus-pinheiro/Documentos/Trabalhos-da-disciplina-de-Sensores-e-Instrumentos-Inteligentes/Trabalho_3/sequencia.csv"

# Carregar o dataset
df_final = pd.read_csv(caminho_arquivo)
print(f"Arquivo carregado: {caminho_arquivo}")

print(f"Shape do dataset: {df_final.shape}")
print("\nPrimeiras linhas do dataset:")
print(df_final.head())
print(f"\nColunas disponíveis: {df_final.columns.tolist()}")

# Verificar quais colunas temos disponíveis
colunas = df_final.columns.tolist()
print(f"\nColunas disponíveis: {colunas}")

# Identificar automaticamente as colunas de features e target
# Procurar por colunas que podem ser features (AccX, AccY, AccZ) ou similares
feature_cols = []
for col in colunas:
    if any(keyword in col.lower() for keyword in ['acc', 'x', 'y', 'z', 'feature', 'attr']):
        feature_cols.append(col)

# Se não encontrou automaticamente, usar as primeiras colunas como features
if len(feature_cols) == 0 and len(colunas) >= 3:
    feature_cols = colunas[:3]  # Usar as 3 primeiras colunas como features

# A última coluna será o target (assumindo que é a coluna de labels)
target_col = colunas[-1]

print(f"\nColunas de features identificadas: {feature_cols}")
print(f"Coluna target identificada: {target_col}")

# Verificar se existem valores NaN
print(f"\nValores NaN no dataset: {df_final.isna().sum().sum()}")

# Remover valores NaN se existirem
df_final = df_final.dropna()

print(f"\nShape após remover NaN: {df_final.shape}")

# Preparar dados para o modelo
X = df_final[feature_cols]
y = df_final[target_col]

print(f"\nDistribuição do target:")
print(y.value_counts())

# Codificar as labels se for necessário (se não forem numéricas)
if y.dtype == 'object':
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"Labels codificadas: {dict(zip(le.classes_, range(len(le.classes_))))}")
else:
    y_encoded = y.values
    le = None
    print("Labels já estão codificadas numericamente")

print(f"\nValores únicos de y: {np.unique(y_encoded)}")

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

print(f"\nDivisão dos dados:")
print(f"Treino: {X_train.shape[0]} amostras")
print(f"Teste: {X_test.shape[0]} amostras")

# Normalizar os dados para alguns modelos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dicionário para armazenar resultados
resultados = {}

print("\n" + "="*60)
print("TREINANDO CLASSIFICADORES")
print("="*60)

# 1. RANDOM FOREST
print("1. Treinando Random Forest...")
inicio = time.time()
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_tempo = time.time() - inicio
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
resultados['Random Forest'] = {'acuracia': rf_accuracy, 'tempo': rf_tempo}
print(f"   ✅ Acurácia: {rf_accuracy:.4f}, Tempo: {rf_tempo:.2f}s")

# 2. GRADIENT BOOSTING
print("2. Treinando Gradient Boosting...")
inicio = time.time()
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)
gb_tempo = time.time() - inicio
gb_pred = gb_model.predict(X_test)
gb_accuracy = accuracy_score(y_test, gb_pred)
resultados['Gradient Boosting'] = {'acuracia': gb_accuracy, 'tempo': gb_tempo}
print(f"   ✅ Acurácia: {gb_accuracy:.4f}, Tempo: {gb_tempo:.2f}s")

# 3. SVM (Support Vector Machine)
print("3. Treinando SVM...")
inicio = time.time()
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train_scaled, y_train)
svm_tempo = time.time() - inicio
svm_pred = svm_model.predict(X_test_scaled)
svm_accuracy = accuracy_score(y_test, svm_pred)
resultados['SVM'] = {'acuracia': svm_accuracy, 'tempo': svm_tempo}
print(f"   ✅ Acurácia: {svm_accuracy:.4f}, Tempo: {svm_tempo:.2f}s")

# 4. K-NN (K-Nearest Neighbors)
print("4. Treinando K-NN...")
inicio = time.time()
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
knn_tempo = time.time() - inicio
knn_pred = knn_model.predict(X_test_scaled)
knn_accuracy = accuracy_score(y_test, knn_pred)
resultados['K-NN'] = {'acuracia': knn_accuracy, 'tempo': knn_tempo}
print(f"   ✅ Acurácia: {knn_accuracy:.4f}, Tempo: {knn_tempo:.2f}s")

# 5. Rede Neural(MLP) 
print("5. Treinando Rede Neural(MLP)...")
inicio = time.time()
mlp_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
mlp_model.fit(X_train_scaled, y_train)
mlp_tempo = time.time() - inicio
mlp_pred = mlp_model.predict(X_test_scaled)
mlp_accuracy = accuracy_score(y_test, mlp_pred)
resultados['Rede Neural(MLP)'] = {'acuracia': mlp_accuracy, 'tempo': mlp_tempo}
print(f"   ✅ Acurácia: {mlp_accuracy:.4f}, Tempo: {mlp_tempo:.2f}s")

# 6. ÁRVORE DE DECISÃO
print("6. Treinando Árvore de Decisão...")
inicio = time.time()
tree_model = DecisionTreeClassifier(max_depth=20, random_state=42)
tree_model.fit(X_train, y_train)
tree_tempo = time.time() - inicio
tree_pred = tree_model.predict(X_test)
tree_accuracy = accuracy_score(y_test, tree_pred)
resultados['Árvore Decisão'] = {'acuracia': tree_accuracy, 'tempo': tree_tempo}
print(f"   ✅ Acurácia: {tree_accuracy:.4f}, Tempo: {tree_tempo:.2f}s")

# COMPARAÇÃO VISUAL
print("\n" + "="*60)
print("COMPARAÇÃO DOS CLASSIFICADORES")
print("="*60)

# Tabela de resultados
df_comparacao = pd.DataFrame(resultados).T
df_comparacao = df_comparacao.sort_values('acuracia', ascending=False)
print("\n📊 TABELA DE RESULTADOS:")
print(df_comparacao.round(4))

# Gráfico de comparação
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Gráfico de acurácia
modelos = df_comparacao.index
acuracias = df_comparacao['acuracia']
cores = ['green' if x == max(acuracias) else 'blue' for x in acuracias]
bars1 = ax1.bar(modelos, acuracias, color=cores, alpha=0.7, edgecolor='black')
ax1.set_title('Comparação de Acurácia dos Classificadores', fontsize=14, fontweight='bold')
ax1.set_ylabel('Acurácia', fontsize=12)
ax1.set_ylim(0, 1.1)
ax1.tick_params(axis='x', rotation=45)
ax1.grid(axis='y', alpha=0.3)

# Adicionar valores nas barras
for bar, acc in zip(bars1, acuracias):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

# Gráfico de tempo
tempos = df_comparacao['tempo']
bars2 = ax2.bar(modelos, tempos, color='red', alpha=0.7, edgecolor='black')
ax2.set_title('Tempo de Treinamento dos Classificadores', fontsize=14, fontweight='bold')
ax2.set_ylabel('Tempo (segundos)', fontsize=12)
ax2.tick_params(axis='x', rotation=45)
ax2.grid(axis='y', alpha=0.3)

# Adicionar valores nas barras
for bar, tempo in zip(bars2, tempos):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{tempo:.2f}s', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('comparacao_classificadores.png', dpi=300, bbox_inches='tight')
plt.show()

# Identificar o melhor modelo
melhor_modelo_nome = df_comparacao.index[0]
melhor_acuracia = df_comparacao.iloc[0]['acuracia']

# Mapear modelos
modelos_dict = {
    'Random Forest': rf_model,
    'Gradient Boosting': gb_model,
    'SVM': svm_model,
    'K-NN': knn_model,
    'Rede Neural(MLP)': mlp_model,
    'Árvore Decisão': tree_model
}

melhor_modelo = modelos_dict[melhor_modelo_nome]

# Salvar o melhor modelo
with open('melhor_modelo.pkl', 'wb') as file:
    pickle.dump(melhor_modelo, file)

# Salvar o label encoder se foi usado
if le is not None:
    with open('label_encoder.pkl', 'wb') as file:
        pickle.dump(le, file)

# Se o melhor modelo usar scaling, salvar o scaler também
if melhor_modelo_nome in ['SVM', 'K-NN', 'Rede Neural(MLP)','Gradient Boosting','Árvore Decisão','Random Forest']:
    with open('scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)

print(f"\n🎉 MELHOR MODELO: {melhor_modelo_nome}")
print(f"📈 Acurácia: {melhor_acuracia:.4f}")
print("💾 Modelo salvo como 'melhor_modelo.pkl'")

# Relatório detalhado do melhor modelo
print(f"\n" + "="*60)
print(f"RELATÓRIO DETALHADO - {melhor_modelo_nome}")
print("="*60)

if melhor_modelo_nome in ['SVM', 'K-NN', 'Rede Neural(MLP)','Gradient Boosting','Árvore Decisão','Random Forest']:
    melhor_pred = melhor_modelo.predict(X_test_scaled)
else:
    melhor_pred = melhor_modelo.predict(X_test)
    
# Obter nomes das classes para o relatório
if le is not None:
    nomes_classes = le.classes_
else:
    # Se não usou LabelEncoder, usar valores únicos do target original
    nomes_classes = [str(cls) for cls in np.unique(y)]

print(classification_report(y_test, melhor_pred, target_names=nomes_classes))

# Matriz de confusão do melhor modelo
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, melhor_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=nomes_classes, yticklabels=nomes_classes)
plt.title(f'Matriz de Confusão - {melhor_modelo_nome}\n(Acurácia: {melhor_acuracia:.4f})', 
          fontsize=14, fontweight='bold')
plt.ylabel('Valor Real')
plt.xlabel('Previsão')
plt.tight_layout()
plt.savefig('matriz_confusao_melhor_modelo.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ Execução concluída com sucesso!")