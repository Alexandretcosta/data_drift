## Grafico de Histograma

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Definindo semente aleatória para reproduzibilidade
np.random.seed(42)
plt.style.use('seaborn-v0_8-whitegrid')

# Simulação dos dados fictícios de Renda Mensal (R$)
# Treino: Média R$ 5.000, Desvio Padrão R$ 1.000
renda_treino = np.random.normal(loc=5000, scale=1000, size=1000)

# Produção (Pós-crise/Inflação): Média cai para R$ 4.200, Desvio Padrão sobe para R$ 1.400
renda_prod = np.random.normal(loc=4200, scale=1400, size=1000)

# Criando figura com dois subplots lado a lado
fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

# 1. Gráfico de Histograma e Densidade de Probabilidade (PDF)
sns.histplot(renda_treino, color='#1f77b4', label=r'$P(X_{\text{treino}})$', 
             kde=True, ax=axes[0], stat="density", alpha=0.4, bins=30)
sns.histplot(renda_prod, color='#ff7f0e', label=r'$P(X_{\text{prod}})$', 
             kde=True, ax=axes[0], stat="density", alpha=0.4, bins=30)

axes[0].set_title('Histograma e Estimativa de Densidade (PDF)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Renda Mensal (R$)', fontsize=10)
axes[0].set_ylabel('Densidade', fontsize=10)
axes[0].legend(title='Distribuição', frameon=True)

# 2. Gráfico de Distribuição Acumulada Empírica (ECDF)
sns.ecdfplot(renda_treino, color='#1f77b4', label=r'CDF Treino $P(X_{\text{treino}})$', ax=axes[1], linewidth=2)
sns.ecdfplot(renda_prod, color='#ff7f0e', label=r'CDF Produção $P(X_{\text{prod}})$', ax=axes[1], linewidth=2)

# Destaque para a Maior Distância Vertical (Estatística D do teste Kolmogorov-Smirnov)
x_eval = 4500
cdf_treino_val = np.mean(renda_treino <= x_eval)
cdf_prod_val = np.mean(renda_prod <= x_eval)

axes[1].vlines(x=x_eval, ymin=cdf_treino_val, ymax=cdf_prod_val, 
               color='red', linestyle='--', linewidth=2, label=r'Distância Máx ($D_{KS}$)')
axes[1].plot([x_eval, x_eval], [cdf_treino_val, cdf_prod_val], 'ro')

axes[1].set_title('Função de Distribuição Acumulada (CDF / K-S)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Renda Mensal (R$)', fontsize=10)
axes[1].set_ylabel('Probabilidade Acumulada', fontsize=10)
axes[1].legend(title='Distribuição', frameon=True)

plt.tight_layout()
plt.show()