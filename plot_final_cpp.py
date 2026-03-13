import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import os
import argparse

INPUT_FILE = "fronteira_pareto_completa.csv"

# ---------------------------------------------------------------------------
# Dados do artigo: NSGA-III (Fig. 3 — Wangsom & Lavangnananda, 2018)
# Valores extraídos manualmente do gráfico de barras (HV normalizado)
# ---------------------------------------------------------------------------
nsga3_data = {
    250:  {"min": 0.016, "max": 0.129, "avg": 0.076},
    500:  {"min": 0.010, "max": 0.048, "avg": 0.028},
    750:  {"min": 0.008, "max": 0.034, "avg": 0.020},
    1000: {"min": 0.002, "max": 0.021, "avg": 0.014},
}

def main():
    parser = argparse.ArgumentParser(
        description="Gera gráfico comparativo de Hypervolume (AEMMT vs NSGA-III)."
    )
    parser.add_argument(
        "-m", "--metodo",
        type=str,
        choices=["Roleta", "Torneio"],
        default="Roleta",
        help="Método de seleção do AEMMT a plotar (padrão: Roleta)"
    )
    args = parser.parse_args()

    METODO_ALVO  = args.metodo
    OUTPUT_IMAGE = f"comparacao_final_hv_{METODO_ALVO.lower()}.png"

    print(f"\n> Método selecionado : {METODO_ALVO}")
    print(f"> Saída              : {OUTPUT_IMAGE}")

    if not os.path.exists(INPUT_FILE):
        print(f"\nERRO: '{INPUT_FILE}' não encontrado.")
        print("Rode './benchmark_app' primeiro.")
        return

    print(f"\nLendo '{INPUT_FILE}'...")
    df = pd.read_csv(INPUT_FILE)

    if "Instance" not in df.columns:
        print("AVISO: coluna 'Instance' não encontrada — criando Instance=1 para compatibilidade.")
        df["Instance"] = 1

    # ---------------------------------------------------------------------------
    # Cálculo de HV por (Size, Instance, Selection, Run)
    # ---------------------------------------------------------------------------
    ref_point = np.array([0.0, 0.0, 0.0])
    ind_hv    = HV(ref_point=ref_point)
    nds       = NonDominatedSorting()

    grouped     = df.groupby(["Size", "Instance", "Selection", "Run"])
    total       = len(grouped)
    results     = {}   # results[size][selection] = [hv, hv, ...]
    count       = 0

    print(f"Calculando HV para {total} grupos (Size × Instance × Selection × Run)...")

    for name, group in grouped:
        size, instance, selection, run = name
        count += 1

        if count % max(1, total // 20) == 0:
            print(f"  {int(count / total * 100):3d}%  ({count}/{total})")

        points = group[["Obj1", "Obj2", "Obj3"]].values

        # Normalização pelo máximo teórico e inversão para minimização
        max_theoretical = size * 100.0
        neg_points      = (points / max_theoretical) * -1.0

        # Filtra fronteira de Pareto (rank 0)
        hv = 0.0
        fronts = nds.do(neg_points)
        if len(fronts) > 0:
            hv = ind_hv(neg_points[fronts[0]])

        results.setdefault(size, {}).setdefault(selection, []).append(hv)

    # ---------------------------------------------------------------------------
    # Verifica se o método escolhido existe, senão usa fallback
    # ---------------------------------------------------------------------------
    sizes_list = [250, 500, 750, 1000]

    has_method = any(
        METODO_ALVO in results.get(s, {}) for s in sizes_list
    )
    if not has_method:
        fallback = "Torneio" if METODO_ALVO == "Roleta" else "Roleta"
        print(f"\nAVISO: dados de '{METODO_ALVO}' não encontrados.")
        print(f"       Usando '{fallback}' como fallback.")
        METODO_ALVO  = fallback
        OUTPUT_IMAGE = f"comparacao_final_hv_{METODO_ALVO.lower()}.png"

    # ---------------------------------------------------------------------------
    # Monta vetores para o gráfico
    # ---------------------------------------------------------------------------
    ammt_vals  = []
    paper_vals = []
    x_labels   = []

    print(f"\n--- RESULTADOS FINAIS ({METODO_ALVO}) ---")
    print(f"{'Tamanho':>8}  {'N amostras':>10}  {'Min':>8}  {'Max':>8}  {'Avg':>8}")
    print("-" * 52)

    for s in sizes_list:
        data = results.get(s, {}).get(METODO_ALVO, [])
        if data:
            my_min = float(np.min(data))
            my_max = float(np.max(data))
            my_avg = float(np.mean(data))
        else:
            my_min = my_max = my_avg = 0.0

        ammt_vals.extend([my_min, my_max, my_avg])

        p = nsga3_data[s]
        paper_vals.extend([p["min"], p["max"], p["avg"]])

        x_labels.extend([f"{s}\nMin", f"{s}\nMax", f"{s}\nAvg"])

        print(f"{s:>8}  {len(data):>10}  {my_min:>8.4f}  {my_max:>8.4f}  {my_avg:>8.4f}")

    # ---------------------------------------------------------------------------
    # Construção do gráfico
    # ---------------------------------------------------------------------------
    group_gap   = 1.5
    current_pos = 0
    plot_pos    = []
    for _ in sizes_list:
        for _ in range(3):
            plot_pos.append(current_pos)
            current_pos += 1
        current_pos += group_gap

    plot_pos = np.array(plot_pos)
    width    = 0.35

    fig, ax = plt.subplots(figsize=(14, 7))

    rects1 = ax.bar(
        plot_pos - width / 2, ammt_vals, width,
        label=f"AEMMT ({METODO_ALVO})",
        color="#4CAF50", edgecolor="black"
    )
    rects2 = ax.bar(
        plot_pos + width / 2, paper_vals, width,
        label="NSGA-III (Wangsom & Lavangnananda, 2018)",
        color="#2196F3", edgecolor="black"
    )

    ax.set_ylabel("Hypervolume (normalizado)", fontsize=12)
    ax.set_title(
        f"AEMMT ({METODO_ALVO}) vs NSGA-III\n"
        f"Métrica: Hypervolume normalizado — 20 instâncias × 30 runs por tamanho",
        fontsize=13
    )
    ax.set_xticks(plot_pos)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    max_val = max(max(ammt_vals), max(paper_vals)) if ammt_vals else 0.1
    ax.set_ylim(0, max_val * 1.18)

    # Rótulos nas barras
    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.annotate(
                f"{h:.3f}",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=8, rotation=90, fontweight="bold"
            )

    autolabel(rects1)
    autolabel(rects2)

    # Separadores visuais entre grupos de tamanho
    for i in range(1, len(sizes_list)):
        sep_x = plot_pos[i * 3 - 1] + (group_gap + 1) / 2
        ax.axvline(sep_x, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"\nSucesso! Gráfico salvo em: '{OUTPUT_IMAGE}'")

if __name__ == "__main__":
    main()