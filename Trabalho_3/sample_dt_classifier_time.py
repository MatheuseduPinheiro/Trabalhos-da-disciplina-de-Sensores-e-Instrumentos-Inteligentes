import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import random
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification

# Verificar se o arquivo existe
caminho_arquivo = "/home/matheus-pinheiro/Documentos/Trabalhos-da-disciplina-de-Sensores-e-Instrumentos-Inteligentes/sequencia.csv"

if not os.path.exists(caminho_arquivo):
    print(f"Arquivo {caminho_arquivo} não encontrado. Criando dados de exemplo...")
    
    # Criar dados de exemplo
    X, y = make_classification(
        n_samples=1000, 
        n_features=2, 
        n_informative=2, 
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Criar DataFrame e salvar como CSV
    dataset = pd.DataFrame({
        'feature1': X[:, 0],
        'feature2': X[:, 1],
        'target': y
    })
    
    # Criar diretório se não existir
    os.makedirs(os.path.dirname(caminho_arquivo), exist_ok=True)
    
    dataset.to_csv(caminho_arquivo, index=False)
    print(f"Dados de exemplo salvos em: {caminho_arquivo}")
else:
    # Carregar dados existentes
    dataset = pd.read_csv(caminho_arquivo)
    print(f"Arquivo carregado: {caminho_arquivo}")

print(f"Shape do dataset: {dataset.shape}")
print(dataset.head())

# Preparar dados
X = dataset.iloc[:, [0, 1]].values  
y = dataset.iloc[:, 2].values  

# Dividir dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42)

# Criar e treinar modelo
model = DecisionTreeClassifier(max_depth=8, random_state=42)
model.fit(X_train, y_train)

print(f"Acurácia do modelo: {model.score(X_test, y_test):.4f}")

# ================================================================
# Teste de performance com perf_counter
# ================================================================
print("\n" + "="*50)
print("TESTE DE PERFORMANCE")
print("="*50)

# perf_counter - inclui tempo de sleep
start = time.perf_counter()

run = 1000
for i in range(run):
    a1 = random.random()
    a2 = random.random()
    y_pred = model.predict([[a1, a2]])

finish = time.perf_counter()
tempo_perf = finish - start

print(f"Tempo pelo perf_counter: {tempo_perf:.4f} segundos")
print(f"Tempo médio por predição: {(tempo_perf/run)*1000:.4f} ms")

# ================================================================
# Teste de performance com process_time
# ================================================================

# process_time - apenas tempo de CPU
start = time.process_time()

for i in range(run):
    a1 = random.random()
    a2 = random.random()
    y_pred = model.predict([[a1, a2]])

finish = time.process_time()
tempo_process = finish - start

print(f"\nTempo pelo process_time: {tempo_process:.4f} segundos")
print(f"Tempo médio por predição: {(tempo_process/run)*1000:.4f} ms")

# ================================================================
# Análise comparativa
# ================================================================
print("\n" + "="*50)
print("ANÁLISE COMPARATIVA")
print("="*50)
print(f"Diferença entre os tempos: {abs(tempo_perf - tempo_process):.4f} segundos")
print(f"Razão perf/process: {tempo_perf/tempo_process:.4f}")

# ================================================================
# Visualização dos dados (opcional)
# ================================================================
def plot_dados_e_decision_boundary():
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Dados originais
    plt.subplot(1, 3, 1)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.6)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Dados de Treinamento')
    plt.colorbar(scatter)
    
    # Plot 2: Decision Boundary
    plt.subplot(1, 3, 2)
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.6)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    
    # Plot 3: Comparação de tempos
    plt.subplot(1, 3, 3)
    metodos = ['perf_counter', 'process_time']
    tempos = [tempo_perf, tempo_process]
    bars = plt.bar(metodos, tempos, color=['skyblue', 'lightcoral'])
    plt.ylabel('Tempo (segundos)')
    plt.title('Comparação de Métodos de Timing')
    
    # Adicionar valores nas barras
    for bar, tempo in zip(bars, tempos):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{tempo:.4f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

# Descomente a linha abaixo para ver a visualização
# plot_dados_e_decision_boundary()

print("\nExecução concluída com sucesso!")