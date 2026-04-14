import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import os
import argparse

INPUT_FILE = "fronteira_pareto_completa.csv"

# ---------------------------------------------------------------------------
# Dados do artigo: NSGA-III (Fig. 3 - Wangsom & Lavangnananda, 2018)
# ---------------------------------------------------------------------------
nsga3_data = {
    250:  {"min": 0.016, "max": 0.129, "avg": 0.076},
    500:  {"min": 0.010, "max": 0.048, "avg": 0.028},
    750:  {"min": 0.008, "max": 0.034, "avg": 0.020},
    1000: {"min": 0.002, "max": 0.021, "avg": 0.014},
}

COLORS = {
    "Torneio":  "#007735",  
    "Roleta":   "#7DB713",  
    "NSGA-III": "#EE7900",  
}

def calc_hv(df):
    """Calcula HV por (Size, Instance, Selection, Run) e retorna dict[size][selection] = [hv,...]"""
    if "Instance" not in df.columns:
        df["Instance"] = 1

    ref_point = np.array([0.0, 0.0, 0.0])
    ind_hv    = HV(ref_point=ref_point)
    nds       = NonDominatedSorting()
    results   = {}

    grouped = df.groupby(["Size", "Instance", "Selection", "Run"])
    total   = len(grouped)
    print(f"Calculando HV para {total} grupos...")

    for count, (name, group) in enumerate(grouped, 1):
        size, instance, selection, run = name

        if count % max(1, total // 20) == 0:
            print(f"  {int(count / total * 100):3d}%  ({count}/{total})")

        points          = group[["Obj1", "Obj2", "Obj3"]].values
        neg_points      = (points / (size * 100.0)) * -1.0

        hv = 0.0
        fronts = nds.do(neg_points)
        if len(fronts) > 0:
            hv = ind_hv(neg_points[fronts[0]])

        results.setdefault(size, {}).setdefault(selection, []).append(hv)

    return results

def main():
    parser = argparse.ArgumentParser(
        description="Comparacao de HV: AEMMT Roleta x AEMMT Torneio x NSGA-III"
    )
    args = parser.parse_args()

    OUTPUT_IMAGE = "comparacao_final_hv_todos.png"

    if not os.path.exists(INPUT_FILE):
        print(f"ERRO: '{INPUT_FILE}' nao encontrado. Rode './benchmark_app' primeiro.")
        return

    print(f"Lendo '{INPUT_FILE}'...")
    df = pd.read_csv(INPUT_FILE)
    df = df.dropna(subset=["Selection"])

    results    = calc_hv(df)
    sizes_list = [250, 500, 750, 1000]
    metodos    = ["Roleta", "Torneio"]
    stats      = ["Min", "Max", "Avg"]

    # ---------------------------------------------------------------------------
    # Vetores de valores para cada algoritmo
    # ---------------------------------------------------------------------------
    vals = {"Roleta": [], "Torneio": [], "NSGA-III": []}

    print(f"\n--- RESULTADOS FINAIS ---")
    print(f"{'Tamanho':>8}  {'Stat':>5}  {'Roleta':>8}  {'Torneio':>8}  {'NSGA-III':>8}")
    print("-" * 50)

    for s in sizes_list:
        for stat in stats:
            # AEMMT Roleta
            data_r = results.get(s, {}).get("Roleta", [])
            v_r = {"Min": np.min, "Max": np.max, "Avg": np.mean}[stat](data_r) if data_r else 0.0

            # AEMMT Torneio
            data_t = results.get(s, {}).get("Torneio", [])
            v_t = {"Min": np.min, "Max": np.max, "Avg": np.mean}[stat](data_t) if data_t else 0.0

            # NSGA-III artigo
            v_n = nsga3_data[s][stat.lower()]

            vals["Roleta"].append(v_r)
            vals["Torneio"].append(v_t)
            vals["NSGA-III"].append(v_n)

            print(f"{s:>8}  {stat:>5}  {v_r:>8.4f}  {v_t:>8.4f}  {v_n:>8.4f}")

    group_gap   = 1.5
    current_pos = 0
    plot_pos    = []
    x_labels    = []

    for s in sizes_list:
        for stat in stats:
            plot_pos.append(current_pos)
            x_labels.append(f"{s}\n{stat}")
            current_pos += 1
        current_pos += group_gap

    plot_pos = np.array(plot_pos)
    width    = 0.26   

    fig, ax = plt.subplots(figsize=(16, 7))

    offsets = [-width, 0, width]
    labels  = ["AEMMT Roleta", "AEMMT Torneio", "NSGA-III (Wangsom & Lavangnananda, 2018)"]
    keys    = ["Roleta", "Torneio", "NSGA-III"]

    rects_list = []
    for offset, key, label in zip(offsets, keys, labels):
        r = ax.bar(
            plot_pos + offset, vals[key], width,
            label=label,
            color=COLORS[key],
            edgecolor="black",
            linewidth=0.6
        )
        rects_list.append(r)

    # Rotulos nas barras
    for rects in rects_list:
        for rect in rects:
            h = rect.get_height()
            if h > 0:
                ax.annotate(
                    f"{h:.3f}",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=7.5, rotation=90, fontweight="bold"
                )

    # Separadores entre grupos de tamanho
    for i in range(1, len(sizes_list)):
        sep_x = plot_pos[i * 3 - 1] + (group_gap + 1) / 2
        ax.axvline(sep_x, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    # Rotulos de tamanho centralizados abaixo dos grupos
    for i, s in enumerate(sizes_list):
        center = np.mean(plot_pos[i*3 : i*3+3])
        ax.text(center, -ax.get_ylim()[1] * 0.07, f"{s} itens",
                ha="center", va="top", fontsize=11, fontweight="bold")

    max_val = max(max(v) for v in vals.values())
    ax.set_ylim(0, max_val * 1.20)
    ax.set_ylabel("Hypervolume (normalizado)", fontsize=12)
    ax.set_title(
        "AEMMT Roleta  vs  AEMMT Torneio  vs  NSGA-III\n"
        "Metrica: Hypervolume normalizado — 20 instancias x 30 runs por tamanho",
        fontsize=13
    )
    ax.set_xticks(plot_pos)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"\nSucesso! Grafico salvo em: '{OUTPUT_IMAGE}'")

if __name__ == "__main__":
    main()