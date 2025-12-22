import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import sys

try:
    df = pd.read_csv("fronteira_pareto_completa.csv")
except FileNotFoundError:
    print("Erro: 'fronteira_pareto_completa.csv' não encontrado.")
    sys.exit(1)

# Dados do Artigo (Fig. 3) para comparação
paper_data = {
    250:  {'min': 0.016, 'max': 0.129, 'avg': 0.076}, 
    500:  {'min': 0.010, 'max': 0.048, 'avg': 0.028},
    750:  {'min': 0.008, 'max': 0.034, 'avg': 0.020},
    1000: {'min': 0.002, 'max': 0.021, 'avg': 0.014}
}


# Estrutura: { size: { 'Roleta': [hv1, hv2...], 'Torneio': [hv1, hv2...] } }
results = {}

print("Calculando Hypervolume (isso pode demorar um pouco)...")

# Agrupa por Tamanho, Seleção e Run
grouped = df.groupby(['Size', 'Selection', 'Run'])

# Prepara o objeto de cálculo HV (Ponto de referência 0,0,0 pois os dados serão invertidos)
ref_point = np.array([0.0, 0.0, 0.0])
ind_hv = HV(ref_point=ref_point)

for name, group in grouped:
    size, selection, run = name
    
    # 1. Pega os objetivos (Matriz N x 3)
    points = group[['Obj1', 'Obj2', 'Obj3']].values
    
    # 2. Normalização 
    # Lucro máximo teórico = size * 100
    max_theorical = size * 100.0
    norm_points = points / max_theorical
    
    # 3. Inverte para Minimização (Pymoo padrão)
    # 1.0 vira -1.0 (melhor), 0.0 vira 0.0 (pior)
    neg_points = norm_points * -1
    
    # 4. Filtra Não-Dominados (Pareto Front)
    nds = NonDominatedSorting()
    fronts = nds.do(neg_points)
    
    # Pega apenas o primeiro front (índice 0)
    if len(fronts) > 0:
        pareto_front = neg_points[fronts[0]]
        
        # 5. Calcula HV
        hv = ind_hv(pareto_front)
    else:
        hv = 0.0
        
    # Salva
    if size not in results: results[size] = {}
    if selection not in results[size]: results[size][selection] = []
    results[size][selection].append(hv)

# --- PLOTAGEM ---
sizes_list = [250, 500, 750, 1000]
x_labels = []
ammt_vals = [] # Vamos plotar 'Roleta' como o principal AMMT para comparar
paper_vals = []

# Preparação dos dados para plotagem (Min, Max, Avg)
# Vamos usar o método 'Roleta' como o representante do AMMT (verde)
plot_pos = []
current_pos = 0
group_gap = 1.0

print("\n--- RESUMO DE PERFORMANCE (AMMT Roleta) ---")

for s in sizes_list:
    # Dados AMMT (Roleta)
    hvs = results[s]['Roleta']
    my_min = np.min(hvs)
    my_max = np.max(hvs)
    my_avg = np.mean(hvs)
    ammt_vals.extend([my_min, my_max, my_avg])
    
    print(f"[{s}] AMMT: {my_avg:.4f} (Min:{my_min:.4f}, Max:{my_max:.4f})")
    
    # Dados Artigo
    p = paper_data[s]
    paper_vals.extend([p['min'], p['max'], p['avg']])
    
    x_labels.extend([f"{s}\nMin", f"{s}\nMax", f"{s}\nAvg"])
    
    for _ in range(3):
        plot_pos.append(current_pos)
        current_pos += 1
    current_pos += group_gap

plot_pos = np.array(plot_pos)
width = 0.35

fig, ax = plt.subplots(figsize=(14, 7))


rects1 = ax.bar(plot_pos - width/2, ammt_vals, width, label='AMMT (Seu Código)', color='#4CAF50', edgecolor='black')
rects2 = ax.bar(plot_pos + width/2, paper_vals, width, label='NSGA-III (Artigo)', color='#2196F3', edgecolor='black')

ax.set_ylabel('Hypervolume (Normalizado)')
ax.set_title('Comparação Final: AMMT vs NSGA-III (30 Execuções)')
ax.set_xticks(plot_pos)
ax.set_xticklabels(x_labels)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.5)


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, rotation=90, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig('comparacao_final_hv.png')
print("\nGráfico gerado: 'comparacao_final_hv.png'")