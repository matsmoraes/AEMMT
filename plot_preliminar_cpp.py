import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Tenta ler o ficheiro
try:
    df = pd.read_csv("resultados_cpp.csv")
except FileNotFoundError:
    print("Erro: 'resultados_cpp.csv' não encontrado.")
    sys.exit(1)

sns.set_theme(style="whitegrid")

# Cria o gráfico de barras
plt.figure(figsize=(12, 6))

# Plota: X=Tamanho, Y=Lucro, Cor=Seleção
# O 'ci="sd"' adiciona a barra de desvio padrão automaticamente
chart = sns.barplot(
    data=df, 
    x="Size", 
    y="BestObjective1", 
    hue="Selection", 
    palette="viridis",
    edgecolor="black",
    errorbar="sd" 
)

plt.title("Análise Preliminar: Melhor Lucro (Objetivo 1) por Método de Seleção", fontsize=14)
plt.xlabel("Tamanho do Problema (Itens)", fontsize=12)
plt.ylabel("Melhor Lucro Encontrado (Média)", fontsize=12)
plt.legend(title="Método de Seleção")

# Adiciona valores nas barras
for container in chart.containers:
    chart.bar_label(container, fmt='%.0f', padding=3, fontsize=9)

plt.tight_layout()
plt.savefig("analise_preliminar_lucro.png")
print("\nGráfico gerado: 'analise_preliminar_lucro.png'")
print("NOTA: Este gráfico mostra LUCRO BRUTO, não Hypervolume.")
```

```bash
