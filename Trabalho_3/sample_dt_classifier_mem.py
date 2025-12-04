import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import plot_tree
from pympler import asizeof

caminho_arquivo = "/home/matheus-pinheiro/Documentos/Trabalhos-da-disciplina-de-Sensores-e-Instrumentos-Inteligentes/Trabalho_3/sequencia.csv"

# ============================================
# CARREGAMENTO DOS DADOS
# ============================================
print("="*60)
print("CARREGAMENTO DOS DADOS")
print("="*60)

# Carregar o dataset
dataset = pd.read_csv(caminho_arquivo)
print(f"✓ Arquivo carregado: {caminho_arquivo}")
print(f"✓ Colunas disponíveis: {list(dataset.columns)}")

# Verificar se as colunas necessárias existem
colunas_necessarias = ['AccX', 'AccY', 'atividade_lb']
for coluna in colunas_necessarias:
    if coluna not in dataset.columns:
        print(f"✗ ERRO: Coluna '{coluna}' não encontrada no dataset!")
        print(f"   Colunas disponíveis: {list(dataset.columns)}")
        exit()

# ============================================
# PREPARAÇÃO DAS FEATURES E TARGET
# ============================================
print("\n" + "="*60)
print("PREPARAÇÃO DOS DADOS")
print("="*60)

# Usar APENAS AccX e AccY como features
X = dataset[['AccX', 'AccY']].values
# Usar atividade_lb como target
y = dataset['atividade_lb'].values

print(f"✓ Features selecionadas: ['AccX', 'AccY']")
print(f"✓ Target selecionado: 'atividade_lb'")
print(f"\nDimensões:")
print(f"  - X (features): {X.shape}")
print(f"  - y (target): {y.shape}")
print(f"\nPrimeiras 5 amostras:")
print(f"  AccX | AccY | atividade_lb")
for i in range(min(5, len(X))):
    print(f"  {X[i,0]:.4f} | {X[i,1]:.4f} | {y[i]}")

# ============================================
# ANÁLISE DOS DADOS
# ============================================
print("\n" + "="*60)
print("ANÁLISE DOS DADOS")
print("="*60)

# Informações estatísticas
print("\nEstatísticas das features:")
print("AccX:")
print(f"  Média: {np.mean(X[:,0]):.4f}")
print(f"  Desvio padrão: {np.std(X[:,0]):.4f}")
print(f"  Mín: {np.min(X[:,0]):.4f}, Máx: {np.max(X[:,0]):.4f}")

print("\nAccY:")
print(f"  Média: {np.mean(X[:,1]):.4f}")
print(f"  Desvio padrão: {np.std(X[:,1]):.4f}")
print(f"  Mín: {np.min(X[:,1]):.4f}, Máx: {np.max(X[:,1]):.4f}")

print(f"\nTarget (atividade_lb):")
print(f"  Número de amostras: {len(y)}")
print(f"  Valores únicos: {np.unique(y)}")
print(f"  Tipo de dados: {y.dtype}")

# Contar frequência de cada classe
unique, counts = np.unique(y, return_counts=True)
print(f"\nDistribuição das classes:")
for valor, frequencia in zip(unique, counts):
    porcentagem = (frequencia / len(y)) * 100
    print(f"  Classe {valor}: {frequencia} amostras ({porcentagem:.1f}%)")

# ============================================
# DECISÃO DO TIPO DE MODELO
# ============================================
print("\n" + "="*60)
print("DECISÃO DO TIPO DE MODELO")
print("="*60)

# Dividir dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42)

# Verificar se é problema de classificação ou regressão
n_unique = len(np.unique(y))
is_classification = n_unique <= 10  # Se tiver até 10 valores únicos, assume classificação

if is_classification:
    print(f"✓ Detectado: Problema de CLASSIFICAÇÃO")
    print(f"  - Número de classes: {n_unique}")
    print(f"  - Classes: {np.unique(y)}")
    
    # Usar DecisionTreeClassifier
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"  - Acurácia do modelo: {score:.4f}")
    print(f"  - Número de amostras no treino: {len(X_train)}")
    print(f"  - Número de amostras no teste: {len(X_test)}")
    
    # Informações das classes
    if hasattr(model, 'classes_'):
        print(f"  - Classes aprendidas: {model.classes_}")
    
else:
    print(f"✓ Detectado: Problema de REGRESSÃO")
    print(f"  - Número de valores únicos: {n_unique}")
    print(f"  - Valores entre {np.min(y):.2f} e {np.max(y):.2f}")
    
    # Usar DecisionTreeRegressor
    model = DecisionTreeRegressor(max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"  - Score R² do modelo: {score:.4f}")
    print(f"  - Número de amostras no treino: {len(X_train)}")
    print(f"  - Número de amostras no teste: {len(X_test)}")

# ============================================
# MEDIÇÃO DA MEMÓRIA DO MODELO
# ============================================
print("\n" + "="*60)
print("MEDIÇÃO DA MEMÓRIA DO MODELO")
print("="*60)

# Medir tamanho do modelo
tamanho_bytes = asizeof.asizeof(model)
tamanho_kb = tamanho_bytes / 1024
tamanho_mb = tamanho_bytes / (1024 * 1024)

print(f"Tamanho total do modelo na memória:")
print(f"  - {tamanho_bytes:,} bytes")
print(f"  - {tamanho_kb:.2f} KB")
print(f"  - {tamanho_mb:.6f} MB")

# Medir também o tamanho dos dados de treino
tamanho_X_train = asizeof.asizeof(X_train)
tamanho_y_train = asizeof.asizeof(y_train)
print(f"\nTamanho dos dados de treino:")
print(f"  - X_train: {tamanho_X_train:,} bytes ({tamanho_X_train/1024:.2f} KB)")
print(f"  - y_train: {tamanho_y_train:,} bytes ({tamanho_y_train/1024:.2f} KB)")

# ============================================
# INFORMAÇÕES DETALHADAS DO MODELO
# ============================================
print("\n" + "="*60)
print("INFORMAÇÕES DETALHADAS DO MODELO")
print("="*60)

print(f"Tipo do modelo: {type(model).__name__}")
print(f"Profundidade da árvore: {model.get_depth()}")
print(f"Número de folhas: {model.get_n_leaves()}")
print(f"Número de features: {model.n_features_in_}")
print(f"Nomes das features: ['AccX', 'AccY']")

if isinstance(model, DecisionTreeClassifier):
    print(f"\nImportância das features:")
    importancias = model.feature_importances_
    for i, importancia in enumerate(importancias):
        print(f"  Feature {i+1} ({['AccX', 'AccY'][i]}): {importancia:.4f}")

# ============================================
# VISUALIZAÇÃO DA ÁRVORE DE DECISÃO
# ============================================
print("\n" + "="*60)
print("VISUALIZAÇÃO DA ÁRVORE DE DECISÃO")
print("="*60)

try:
    plt.figure(figsize=(15, 10))
    
    if isinstance(model, DecisionTreeClassifier):
        # Para classificador
        plot_tree(model, 
                  feature_names=['AccX', 'AccY'],
                  class_names=[f'Classe_{int(c)}' for c in np.unique(y)],
                  filled=True, 
                  rounded=True,
                  fontsize=10,
                  proportion=True)
    else:
        # Para regressor
        plot_tree(model, 
                  feature_names=['AccX', 'AccY'],
                  filled=True, 
                  rounded=True,
                  fontsize=10,
                  proportion=True)
    
    plt.title(f"Árvore de Decisão - Modelo: {type(model).__name__}\nFeatures: AccX e AccY | Target: atividade_lb", 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Salvar imagem
    output_path = caminho_arquivo.replace('.csv', '_decision_tree_accx_accy.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Árvore de decisão salva em: {output_path}")
    
    # Mostrar gráfico
    plt.show()
    
except Exception as e:
    print(f"✗ Erro ao gerar visualização: {e}")

# ============================================
# VISUALIZAÇÃO DOS DADOS
# ============================================
print("\n" + "="*60)
print("VISUALIZAÇÃO DOS DADOS (SCATTER PLOT)")
print("="*60)

try:
    plt.figure(figsize=(12, 8))
    
    # Criar scatter plot colorido por classe
    if len(np.unique(y)) <= 10:  # Se for classificação
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='tab10', alpha=0.6, s=30)
        plt.colorbar(scatter, label='Classe (atividade_lb)')
    else:  # Se for regressão
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.6, s=30)
        plt.colorbar(scatter, label='Valor (atividade_lb)')
    
    plt.xlabel('AccX', fontsize=12)
    plt.ylabel('AccY', fontsize=12)
    plt.title('Distribuição dos Dados: AccX vs AccY\nColorido por atividade_lb', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Salvar scatter plot
    scatter_path = caminho_arquivo.replace('.csv', '_scatter_accx_accy.png')
    plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
    print(f"✓ Scatter plot salvo em: {scatter_path}")
    
    plt.show()
    
except Exception as e:
    print(f"✗ Erro ao gerar scatter plot: {e}")

print("\n" + "="*60)
print("PROCESSAMENTO CONCLUÍDO!")
print("="*60)
print(f"✓ Features usadas: AccX, AccY")
print(f"✓ Target: atividade_lb")
print(f"✓ Tipo de modelo: {type(model).__name__}")
print(f"✓ Tamanho do modelo: {tamanho_mb:.6f} MB")
print("="*60)