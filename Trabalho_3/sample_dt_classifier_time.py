import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# CORREÇÃO: dataset com um "t" apenas
dataset = pd.read_csv("sequencia.csv")

X = dataset[['AccX', 'AccY', 'AccZ']]  # Usar lista de strings com double brackets
y = dataset['atividade_lb']


# Dividir dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42)

print(f"\nDivisão dos dados:")
print(f"Treino: X={X_train.shape}, y={y_train.shape}")
print(f"Teste:  X={X_test.shape}, y={y_test.shape}")

# Criar e treinar modelo
model = DecisionTreeClassifier(max_depth=8, random_state=42)
model.fit(X_train, y_train)

print(f"\nAcurácia do modelo: {model.score(X_test, y_test):.4f}")

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
    a3 = random.random()  # Terceira feature
    y_pred = model.predict([[a1, a2, a3]])  # 3 features

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
    a3 = random.random()  # Terceira feature
    y_pred = model.predict([[a1, a2, a3]])  # 3 features

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
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: AccX vs AccY
    scatter1 = axes[0].scatter(X['AccX'], X['AccY'], c=y, cmap='viridis', alpha=0.6)
    axes[0].set_xlabel('AccX')
    axes[0].set_ylabel('AccY')
    axes[0].set_title('AccX vs AccY')
    plt.colorbar(scatter1, ax=axes[0])
    
    # Plot 2: AccX vs AccZ
    scatter2 = axes[1].scatter(X['AccX'], X['AccZ'], c=y, cmap='viridis', alpha=0.6)
    axes[1].set_xlabel('AccX')
    axes[1].set_ylabel('AccZ')
    axes[1].set_title('AccX vs AccZ')
    plt.colorbar(scatter2, ax=axes[1])
    
    # Plot 3: Comparação de tempos
    metodos = ['perf_counter', 'process_time']
    tempos = [tempo_perf, tempo_process]
    bars = axes[2].bar(metodos, tempos, color=['skyblue', 'lightcoral'])
    axes[2].set_ylabel('Tempo (segundos)')
    axes[2].set_title('Comparação de Métodos de Timing')
    
    # Adicionar valores nas barras
    for bar, tempo in zip(bars, tempos):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{tempo:.4f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

# Descomente a linha abaixo para ver a visualização
# plot_dados_e_decision_boundary()

print("\nExecução concluída com sucesso!")