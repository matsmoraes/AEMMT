import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import sys
import os

INPUT_FILE = "fronteira_pareto_completa.csv"

# Define qual método tentar plotar primeiro
METODO_ALVO = 'Roleta' 

# O nome do arquivo de saída será baseado no método alvo
OUTPUT_IMAGE = f"comparacao_final_hv_{METODO_ALVO.lower()}.png"

# ==============================================================================

# --- DADOS DO ARTIGO (E-NSGA-III - Fig. 3) ---
paper_data = {
    250:  {'min': 0.016, 'max': 0.129, 'avg': 0.076}, 
    500:  {'min': 0.010, 'max': 0.048, 'avg': 0.028},
    750:  {'min': 0.008, 'max': 0.034, 'avg': 0.020},
    1000: {'min': 0.002, 'max': 0.021, 'avg': 0.014}
}

# --- FUNÇÃO PRINCIPAL ---
def main():
    # 1. Verifica se o arquivo existe
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

    # Estruturas para cálculo do HV
    results = {}
    ref_point = np.array([0.0, 0.0, 0.0]) # Ponto de referência (0,0,0) para dados normalizados/invertidos
    ind_hv = HV(ref_point=ref_point)
    nds = NonDominatedSorting()

    grouped = df.groupby(['Size', 'Selection', 'Run'])
    total_groups = len(grouped)
    count = 0

    print(f"Calculando Hypervolume para {total_groups} execuções...")

    # 2. Loop de processamento dos dados
    for name, group in grouped:
        size, selection, run = name
        count += 1
        
        if count % (total_groups // 10 + 1) == 0:
            print(f"Progresso: {int(count/total_groups*100)}%...")

        # Extrai os objetivos
        points = group[['Obj1', 'Obj2', 'Obj3']].values
        
        # --- NORMALIZAÇÃO E INVERSÃO ---
        # Divide pelo máximo teórico (Size * 100) para ficar entre 0 e 1
        max_theoretical = size * 100.0
        norm_points = points / max_theoretical
        
        # Inverte para minimização (padrão pymoo): Melhores valores ficam próximos de -1
        neg_points = norm_points * -1

        # --- FILTRO DE PARETO ---
        fronts = nds.do(neg_points)
        hv = 0.0
        if len(fronts) > 0:
            # Pega apenas a primeira fronteira (Rank 0)
            pareto_front = neg_points[fronts[0]]
            hv = ind_hv(pareto_front)

        # Armazena o resultado
        if size not in results: results[size] = {}
        if selection not in results[size]: results[size][selection] = []
        results[size][selection].append(hv)

    # 3. Preparação para Plotagem
    print("\nGerando Gráfico...")
    sizes_list = [250, 500, 750, 1000]
    ammt_vals = [] 
    paper_vals = []
    x_labels = []
    
    metodo_escolhido = METODO_ALVO
    
    # Verifica se existe dados do método escolhido, senão tenta fallback
    has_method = False
    for s in sizes_list:
        if s in results and metodo_escolhido in results[s]:
            has_method = True
            break
    
    if not has_method:
        print(f"AVISO: Dados de '{metodo_escolhido}' não encontrados no CSV.")
        # Tenta encontrar o outro método (se Roleta falhar, tenta Torneio e vice-versa)
        outro_metodo = 'Torneio' if metodo_escolhido == 'Roleta' else 'Roleta'
        print(f"-> Tentando usar '{outro_metodo}' como fallback.")
        metodo_escolhido = outro_metodo

    print(f"\n--- RESULTADOS FINAIS ({metodo_escolhido}) ---")
    
    for s in sizes_list:
        # Dados do AEMMT
        if s in results and metodo_escolhido in results[s]:
            data = results[s][metodo_escolhido]
            my_min = np.min(data)
            my_max = np.max(data)
            my_avg = np.mean(data)
        else:
            my_min, my_max, my_avg = 0, 0, 0
        
        ammt_vals.extend([my_min, my_max, my_avg])
        
        # Dados do Artigo (NSGA-III)
        p = paper_data[s]
        paper_vals.extend([p['min'], p['max'], p['avg']])
        
        x_labels.extend([f"{s}\nMin", f"{s}\nMax", f"{s}\nAvg"])
        
        print(f"[{s} Itens] AEMMT: {my_avg:.4f} (Min:{my_min:.4f} Max:{my_max:.4f})")

    # 4. Construção do Gráfico
    group_gap = 1.5
    current_pos = 0
    plot_pos = []
    
    for _ in sizes_list:
        for _ in range(3):
            plot_pos.append(current_pos)
            current_pos += 1
        current_pos += group_gap
    
    plot_pos = np.array(plot_pos)
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 7))

    # Barra Verde (AMMT)
    rects1 = ax.bar(plot_pos - width/2, ammt_vals, width, 
                    label=f'AEMMT ({metodo_escolhido})', color='#4CAF50', edgecolor='black')
    
    # Barra Azul (NSGA-III Paper)
    rects2 = ax.bar(plot_pos + width/2, paper_vals, width, 
                    label='NSGA-III (Artigo)', color='#2196F3', edgecolor='black')

    ax.set_ylabel('Hypervolume (Normalizado)')
    ax.set_title(f'Comparação: AEMMT ({metodo_escolhido}) vs NSGA-III\n(Métrica: Hypervolume Normalizado)')
    ax.set_xticks(plot_pos)
    ax.set_xticklabels(x_labels)
    ax.legend(loc='upper left')
    
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Ajuste do limite Y
    max_val = max(max(ammt_vals), max(paper_vals)) if ammt_vals else 0.1
    ax.set_ylim(0, max_val * 1.15)

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
    
    # Salva a imagem
    nome_arquivo_final = OUTPUT_IMAGE
    if metodo_escolhido != METODO_ALVO:
        nome_arquivo_final = f"comparacao_final_hv_{metodo_escolhido.lower()}.png"
        
    plt.savefig(nome_arquivo_final)
    print(f"\nSucesso! Gráfico salvo em: '{nome_arquivo_final}'")

if __name__ == "__main__":
    main()