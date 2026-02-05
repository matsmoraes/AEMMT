import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

INPUT_FILE = "fronteira_pareto_completa.csv"
OUTPUT_IMAGE = "analise_objetivos_individuais.png"

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"ERRO: '{INPUT_FILE}' não encontrado.")
        return

    print(f"Lendo '{INPUT_FILE}'...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except Exception as e:
        print(f"Erro ao ler CSV: {e}")
        return
    
    print("Processando melhores valores por execução...")
    
    best_per_run = df.groupby(['Size', 'Selection', 'Run'])[['Obj1', 'Obj2', 'Obj3']].max().reset_index()

    df_melted = best_per_run.melt(
        id_vars=['Size', 'Selection', 'Run'], 
        value_vars=['Obj1', 'Obj2', 'Obj3'],
        var_name='Objetivo', 
        value_name='Lucro'
    )

    # --- PLOTAGEM ---
    print("Gerando gráficos...")
    sns.set_theme(style="whitegrid")

    
    sizes = sorted(df_melted['Size'].unique())
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, size in enumerate(sizes):
        ax = axes[i]
        data_subset = df_melted[df_melted['Size'] == size]
        
        # Gráfico de Barras: X = Objetivo, Y = Lucro, Cor = Método de Seleção
        sns.barplot(
            data=data_subset,
            x="Objetivo",
            y="Lucro",
            hue="Selection",
            palette="viridis",
            errorbar="sd", 
            ax=ax,
            edgecolor="black"
        )
        
        ax.set_title(f"Tamanho do Problema: {size} Itens", fontsize=14, fontweight='bold')
        ax.set_ylabel("Melhor Lucro (Média das Runs)", fontsize=12)
        ax.set_xlabel("")
        ax.legend(title="Método", loc='upper left', framealpha=0.9)
        

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"\nSucesso! Gráfico salvo em: '{OUTPUT_IMAGE}'")
    
    print("\n=== RESUMO ESTATÍSTICO (Média dos Melhores) ===")
    summary = best_per_run.groupby(['Size', 'Selection'])[['Obj1', 'Obj2', 'Obj3']].agg(['mean', 'std'])
    print(summary)

if __name__ == "__main__":
    main()