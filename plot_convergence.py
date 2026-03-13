import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

INPUT_FILE   = "evolucao_fitness.csv"
OUTPUT_IMAGE = "analise_convergencia.png"

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"ERRO: '{INPUT_FILE}' nao encontrado.")
        print("Rode './benchmark_app' primeiro.")
        return

    print(f"Lendo '{INPUT_FILE}'...")
    df = pd.read_csv(INPUT_FILE)

    df = df.dropna(subset=["Selection"])

    if "Instance" not in df.columns:
        print("AVISO: coluna 'Instance' nao encontrada - assumindo CSV no formato antigo.")

    sns.set_theme(style="whitegrid")

    metodos = df["Selection"].unique()
    fig, axes = plt.subplots(1, len(metodos), figsize=(7 * len(metodos), 6), sharey=False)
    if len(metodos) == 1:
        axes = [axes]

    print(f"Gerando graficos de convergencia ({len(metodos)} metodo(s))...")

    for i, metodo in enumerate(metodos):
        ax = axes[i]
        subset = df[df["Selection"] == metodo]

        sns.lineplot(
            data=subset,
            x="Generation",
            y="BestFit",
            hue="Size",
            palette="viridis",
            ax=ax,
            linewidth=2.0
        )

        n_inst = subset["Instance"].nunique() if "Instance" in subset.columns else 1
        n_runs = subset["Run"].nunique()
        ax.set_title(
            f"Convergencia: Metodo {metodo}\n"
            f"(media sobre {n_inst} instancias x {n_runs} runs = {n_inst * n_runs} amostras)",
            fontsize=13, fontweight="bold"
        )
        ax.set_xlabel("Geracoes", fontsize=12)
        ax.set_ylabel("Melhor Fitness Total (soma dos 3 objetivos)", fontsize=12)
        ax.legend(title="Itens", loc="upper left", framealpha=0.9)
        ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"\nSucesso! Grafico salvo em: '{OUTPUT_IMAGE}'")

if __name__ == "__main__":
    main()