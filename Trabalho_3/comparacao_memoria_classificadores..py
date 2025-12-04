import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from pympler import asizeof
import matplotlib.pyplot as plt

caminho_arquivo = "/home/matheus-pinheiro/Documentos/Trabalhos-da-disciplina-de-Sensores-e-Instrumentos-Inteligentes/Trabalho_3/sequencia.csv"

# Carregar dados
df = pd.read_csv(caminho_arquivo)
print(f"✓ Dataset carregado: {caminho_arquivo}")

# Usar AccX e AccY como features, atividade_lb como target
feature_cols = ['AccX', 'AccY'] if 'AccX' in df.columns and 'AccY' in df.columns else df.columns[:2]
target_col = 'atividade_lb' if 'atividade_lb' in df.columns else df.columns[-1]

X = df[feature_cols].values
y = df[target_col].values

# Codificar se necessário
if y.dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)

# Dividir dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalizar para alguns modelos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================
# APENAS OS 6 MODELOS SOLICITADOS
# ============================================
modelos_para_comparar = {
    'Árvore de Decisão': DecisionTreeClassifier(max_depth=10, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42),
    'K-NN': KNeighborsClassifier(n_neighbors=5),
    'Rede Neural (MLP)': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
}

# ============================================
# MEDIR MEMÓRIA DE CADA MODELO
# ============================================
resultados_memoria = []

print("="*70)
print("MEDINDO MEMÓRIA DOS 6 CLASSIFICADORES")
print("="*70)

for nome, modelo in modelos_para_comparar.items():
    print(f"\n🔧 Treinando: {nome}")
    
    # Verificar se precisa de normalização
    if nome in ['SVM', 'K-NN', 'Rede Neural (MLP)']:
        X_tr = X_train_scaled
    else:
        X_tr = X_train
    
    modelo.fit(X_tr, y_train)
    
    tamanho_bytes = asizeof.asizeof(modelo)
    tamanho_mb = tamanho_bytes / (1024 * 1024)
    
    resultados_memoria.append({
        'Modelo': nome,
        'Tamanho_Bytes': tamanho_bytes,
        'Tamanho_MB': tamanho_mb
    })
    
    print(f"  ✓ Memória: {tamanho_mb:.6f} MB ({tamanho_bytes:,} bytes)")

df_resultados = pd.DataFrame(resultados_memoria)

# ============================================
# GRÁFICO DE PIZZA AJUSTADO (SEM TUDO COLADO)
# ============================================

plt.figure(figsize=(10, 10))

df_resultados = df_resultados.sort_values('Tamanho_Bytes', ascending=False)
labels = df_resultados['Modelo'].tolist()
sizes = df_resultados['Tamanho_Bytes'].tolist()

cores = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
explode = [0.08] * len(df_resultados)

wedges, texts, autotexts = plt.pie(
    sizes,
    labels=None,
    colors=cores,
    explode=explode,
    autopct=lambda pct: f'{pct:.1f}%',
    startangle=90,
    pctdistance=0.75,
    textprops={'fontsize': 12, 'fontweight': 'bold'},
    wedgeprops={'linewidth': 1.5, 'edgecolor': 'white'}
)

plt.axis('equal')

plt.title(
    'Comparação de Espaço em Memória — 6 Classificadores',
    fontsize=16,
    fontweight='bold',
    pad=20
)

legend_labels = [
    f"{row['Modelo']}: {row['Tamanho_MB']:.6f} MB"
    for _, row in df_resultados.iterrows()
]

plt.legend(
    wedges,
    legend_labels,
    title="Modelos e Memória",
    loc="center left",
    bbox_to_anchor=(1.05, 0.5),
    fontsize=11,
    title_fontsize=12,
    frameon=True
)

plt.tight_layout()
plt.show()

# ============================================
# TABELA COMPARATIVA DETALHADA
# ============================================
print("\n" + "="*70)
print("TABELA COMPARATIVA - ESPAÇO EM MEMÓRIA")
print("="*70)

print(f"\n{'Classificador':<25} {'Tamanho (MB)':<18} {'Tamanho (KB)':<18} {'Bytes':<25}")
print("-" * 90)

for _, row in df_resultados.sort_values('Tamanho_MB', ascending=False).iterrows():
    tamanho_kb = row['Tamanho_Bytes'] / 1024
    print(f"{row['Modelo']:<25} {row['Tamanho_MB']:<18.6f} {tamanho_kb:<18.2f} {row['Tamanho_Bytes']:<25,}")

print("-" * 90)

memoria_total = sum(sizes)
print(f"\n📊 ESTATÍSTICAS:")
print(f"  • Memória total: {memoria_total/(1024*1024):.4f} MB")
print(f"  • Média por modelo: {df_resultados['Tamanho_MB'].mean():.6f} MB")
print(f"  • Modelo mais leve: {df_resultados.loc[df_resultados['Tamanho_MB'].idxmin()]['Modelo']} ({df_resultados['Tamanho_MB'].min():.6f} MB)")
print(f"  • Modelo mais pesado: {df_resultados.loc[df_resultados['Tamanho_MB'].idxmax()]['Modelo']} ({df_resultados['Tamanho_MB'].max():.6f} MB)")
print(f"  • Diferença: {df_resultados['Tamanho_MB'].max()/df_resultados['Tamanho_MB'].min():.1f}x")

print(f"\n🔍 COMPARAÇÃO RELATIVA (em relação ao mais leve):")
modelo_mais_leve = df_resultados.loc[df_resultados['Tamanho_MB'].idxmin()]

for _, row in df_resultados.iterrows():
    if row['Modelo'] != modelo_mais_leve['Modelo']:
        vezes_maior = row['Tamanho_MB'] / modelo_mais_leve['Tamanho_MB']
        print(f"  • {row['Modelo']:20} é {vezes_maior:.1f}x maior que {modelo_mais_leve['Modelo']}")

print("\n" + "="*70)
print("COMPARAÇÃO CONCLUÍDA!")
print("="*70)
