import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import sys
import os

# --- CONFIGURAÇÕES ---
INPUT_FILE = "fronteira_pareto_completa.csv"
OUTPUT_IMAGE = "comparacao_final_hv.png"

# --- DADOS DO ARTIGO (E-NSGA-III - Fig. 3) ---
# Valores aproximados extraídos visualmente do paper
paper_data = {
    250:  {'min': 0.016, 'max': 0.129, 'avg': 0.076}, 
    500:  {'min': 0.010, 'max': 0.048, 'avg': 0.028},
    750:  {'min': 0.008, 'max': 0.034, 'avg': 0.020},
    1000: {'min': 0.002, 'max': 0.021, 'avg': 0.014}
}

# --- FUNÇÃO PRINCIPAL ---
def main():
    # 1. Verifica e Carrega o CSV
    if not os.path.exists(INPUT_FILE):
        print(f"ERRO: O arquivo '{INPUT_FILE}' não foi encontrado.")
        return

    print(f"Lendo '{INPUT_FILE}'... (Isso pode levar alguns segundos)")
    try:
        df = pd.read_csv(INPUT_FILE)
    except Exception as e:
        print(f"Erro ao ler CSV: {e}")
        return

    # Estrutura para guardar resultados: { tamanho: { 'Roleta': [], 'Torneio': [] } }
    results = {}

    # 2. Configura o Cálculo de Hypervolume
    # Ponto de referência (0,0,0) pois inverteremos os dados para minimização
    ref_point = np.array([0.0, 0.0, 0.0])
    ind_hv = HV(ref_point=ref_point)
    nds = NonDominatedSorting()

    # 3. Processa cada Grupo (Size, Selection, Run)
    # Agrupa os dados para processar cada execução individualmente
    grouped = df.groupby(['Size', 'Selection', 'Run'])
    total_groups = len(grouped)
    count = 0

    print(f"Processando {total_groups} execuções...")

    for name, group in grouped:
        size, selection, run = name
        count += 1
        
        # Feedback de progresso a cada 10%
        if count % (total_groups // 10 + 1) == 0:
            print(f"Progresso: {int(count/total_groups*100)}%...")

        # Extrai os objetivos (Lucros)
        points = group[['Obj1', 'Obj2', 'Obj3']].values

        # --- NORMALIZAÇÃO ---
        # Lucro Máximo Teórico = Num_Itens * 100
        max_theoretical = size * 100.0
        norm_points = points / max_theoretical

        # --- FILTRAGEM DE PARETO ---
        # Pymoo trabalha com MINIMIZAÇÃO. Invertemos os sinais.
        # Melhor fitness = -1.0, Pior = 0.0
        neg_points = norm_points * -1

        # Encontra fronteiras não dominadas
        fronts = nds.do(neg_points)
        
        hv = 0.0
        if len(fronts) > 0:
            # Pega apenas a primeira fronteira (índice 0)
            pareto_front = neg_points[fronts[0]]
            # Calcula HV
            hv = ind_hv(pareto_front)

        # Guarda o resultado
        if size not in results: results[size] = {}
        if selection not in results[size]: results[size][selection] = []
        results[size][selection].append(hv)

    # 4. Prepara Dados para o Gráfico
    print("\nGerando Gráfico...")
    
    sizes_list = [250, 500, 750, 1000]
    
    # Listas "achatadas" para o matplotlib
    ammt_vals = [] 
    paper_vals = []
    x_labels = []
    
    # Escolha qual método do seu código será comparado (Torneio costuma ser melhor em escala)
    metodo_escolhido = 'Torneio' 
    # Verifica se existe dados de Torneio, senão usa Roleta
    # (Adaptação caso seu CSV só tenha um dos dois)
    has_torneio = False
    for s in sizes_list:
        if s in results and 'Torneio' in results[s]:
            has_torneio = True
            break
    
    if not has_torneio:
        metodo_escolhido = 'Roleta'

    print(f"\n--- RESULTADOS FINAIS ({metodo_escolhido}) ---")
    
    for s in sizes_list:
        if s in results and metodo_escolhido in results[s]:
            data = results[s][metodo_escolhido]
            my_min = np.min(data)
            my_max = np.max(data)
            my_avg = np.mean(data)
        else:
            my_min, my_max, my_avg = 0, 0, 0
        
        ammt_vals.extend([my_min, my_max, my_avg])
        
        # Dados do Paper
        p = paper_data[s]
        paper_vals.extend([p['min'], p['max'], p['avg']])
        
        # Labels para o eixo X
        x_labels.extend([f"{s}\nMin", f"{s}\nMax", f"{s}\nAvg"])
        
        print(f"[{s} Itens] AMMT: {my_avg:.4f} (Min:{my_min:.4f} Max:{my_max:.4f})")

    # 5. Plotagem
    # Posições das barras
    group_gap = 1.5 # Espaço entre grupos de tamanho
    current_pos = 0
    plot_pos = []
    
    for _ in sizes_list:
        for _ in range(3): # Min, Max, Avg
            plot_pos.append(current_pos)
            current_pos += 1
        current_pos += group_gap
    
    plot_pos = np.array(plot_pos)
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 7))

    # Barra Verde (AMMT)
    rects1 = ax.bar(plot_pos - width/2, ammt_vals, width, 
                    label=f'AMMT ({metodo_escolhido})', color='#4CAF50', edgecolor='black')
    
    # Barra Azul (NSGA-III Paper)
    rects2 = ax.bar(plot_pos + width/2, paper_vals, width, 
                    label='NSGA-III (Artigo)', color='#2196F3', edgecolor='black')

    # Estilização
    ax.set_ylabel('Hypervolume (Normalizado)')
    ax.set_title(f'Comparação Final: AMMT ({metodo_escolhido}) vs NSGA-III\n(Métrica: Hypervolume Normalizado)')
    ax.set_xticks(plot_pos)
    ax.set_xticklabels(x_labels)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Ajuste de limite Y para caber os números
    max_val = max(max(ammt_vals), max(paper_vals))
    ax.set_ylim(0, max_val * 1.15)

    # Função para colocar valores nas barras
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, rotation=90, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE)
    print(f"\nSucesso! Gráfico salvo em: '{OUTPUT_IMAGE}'")

if __name__ == "__main__":
    main()