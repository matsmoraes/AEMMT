import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# --- CONFIGURAÇÕES ---
INPUT_FILE = "evolucao_fitness.csv"
OUTPUT_IMAGE = "analise_convergencia.png"

# --- FUNÇÃO PRINCIPAL ---
def main():
    if not os.path.exists(INPUT_FILE):
        print(f"ERRO: O arquivo '{INPUT_FILE}' não foi encontrado.")
        print("Certifique-se de ter rodado o benchmark C++ ('./benchmark_app') primeiro.")
        return

    print(f"Lendo '{INPUT_FILE}'...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except Exception as e:
        print(f"Erro ao ler CSV: {e}")
        return

    sns.set_theme(style="whitegrid")
    
    # Cria uma figura com subplots para separar por Método de Seleção (se houver mais de um)
    metodos = df['Selection'].unique()
    
    fig, axes = plt.subplots(1, len(metodos), figsize=(7 * len(metodos), 6), sharey=False)
    
    if len(metodos) == 1:
        axes = [axes]

    print("Gerando gráficos de convergência...")

    for i, metodo in enumerate(metodos):
        ax = axes[i]
        data_subset = df[df['Selection'] == metodo]
        
        # Plota a linha de evolução
        # X = Geração, Y = BestFit (Melhor Fitness)
        # Hue = Size (Uma cor para cada tamanho de problema)
        sns.lineplot(
            data=data_subset, 
            x="Generation", 
            y="BestFit", 
            hue="Size", 
            palette="viridis", 
            ax=ax,
            linewidth=2.0
        )
        
        ax.set_title(f"Convergência: Método {metodo}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Gerações", fontsize=12)
        ax.set_ylabel("Melhor Lucro (Média 30 Runs)", fontsize=12)
        ax.legend(title="Itens")
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"\nSucesso! Gráfico de convergência salvo em: '{OUTPUT_IMAGE}'")

if __name__ == "__main__":
    main()