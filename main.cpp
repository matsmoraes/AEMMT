#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <string>
#include <chrono>

using namespace std;

// ============================================================================
// CONFIGURAÇÕES GLOBAIS
// ============================================================================
const int NUM_OBJECTIVES = 3;      // Problema com 3 objetivos (Mochilas)
const int SUBPOP_COUNT = 3;        // Uma sub-população dedicada para cada objetivo
const double CAPACITY_RATIO = 0.5; // Capacidade da mochila é 50% do peso total dos itens

// ============================================================================
// ESTRUTURAS DE DADOS
// ============================================================================

struct Item {
    int id;
    double weight;
    vector<double> profits; 
    vector<double> ratios;  
};

struct Individual {
    vector<bool> chromosome; // Representação Binária: 1 = Item na mochila, 0 = Fora
    vector<double> fitness;  // Vetor com o lucro de cada uma das 3 mochilas
    double total_weight;
    bool valid;

    Individual(int num_items = 0) {
        if (num_items > 0) {
            chromosome.resize(num_items, false);
            fitness.resize(NUM_OBJECTIVES, 0.0);
        }
        total_weight = 0.0;
        valid = false;
    }
};

// ============================================================================
// CLASSE DO SOLUCIONADOR (AEMMT)
// ============================================================================
class AMMTSolver {
private:
    int num_items;
    int pop_size;
    int selection_method; 
    double mutation_rate;
    double elitism_rate;
    double max_capacity;

    vector<Item> items;
    vector<Individual> population;
    mt19937 rng;

    // Ficheiro para log de evolução
    ofstream* log_evolution;
    int current_run_id;
    string current_sel_name;
    int current_size;

public:
    AMMTSolver(int items_n, int sel_method, double mut_rate, double elit_rate) 
        : num_items(items_n), selection_method(sel_method), 
          mutation_rate(mut_rate), elitism_rate(elit_rate) {
        
        unsigned seed = chrono::system_clock::now().time_since_epoch().count();
        rng.seed(seed);
            
        // 90 = 3 sub-populações de 30 indivíduos cada.
        // Isso evita "sobras" no loop de geração e mantém paridade com os 92 do artigo.
        pop_size = 90; 
        
        log_evolution = nullptr;
        generate_instance();
    }

    void set_logging(ofstream* log_file, int size, string sel, int run) {
        log_evolution = log_file;
        current_size = size;
        current_sel_name = sel;
        current_run_id = run;
    }

    // Gera instância aleatória (Pesos e Lucros entre 10 e 100)
    void generate_instance() {
        items.clear();
        double total_weight_all = 0;
        for (int i = 0; i < num_items; ++i) {
            Item it;
            it.id = i;
            it.weight = uniform_real_distribution<>(10.0, 100.0)(rng);
            total_weight_all += it.weight;
            for (int o = 0; o < NUM_OBJECTIVES; ++o) {
                double profit = uniform_real_distribution<>(10.0, 100.0)(rng);
                it.profits.push_back(profit);
                // Pré-calcula a razão (Lucro / Peso) para usar no Greedy Repair
                it.ratios.push_back(profit / it.weight);
            }
            items.push_back(it);
        }
        max_capacity = total_weight_all * CAPACITY_RATIO;
    }

    
    // Operador de Reparo (Greedy Repair)
    void evaluate_and_repair(Individual& ind) {
        ind.total_weight = 0;
        fill(ind.fitness.begin(), ind.fitness.end(), 0.0);

        // 1. Calcula peso e fitness inicial
        for (int i = 0; i < num_items; ++i) {
            if (ind.chromosome[i]) {
                ind.total_weight += items[i].weight;
                for (int o = 0; o < NUM_OBJECTIVES; ++o) {
                    ind.fitness[o] += items[i].profits[o];
                }
            }
        }

        // 2. Se excedeu a capacidade, aplica o Reparo Guloso
        if (ind.total_weight > max_capacity) {
            vector<int> items_in_bag;
            for (int i = 0; i < num_items; ++i) {
                if (ind.chromosome[i]) items_in_bag.push_back(i);
            }
            
            // Ordena itens presentes pelo "Pior Ratio" (Menor lucro por peso)
            sort(items_in_bag.begin(), items_in_bag.end(), [&](int a, int b) {
                double ratio_a = 0, ratio_b = 0;
                // Usa a soma dos ratios dos 3 objetivos para decidir quem sai
                for(int o=0; o<NUM_OBJECTIVES; ++o) {
                    ratio_a += items[a].ratios[o];
                    ratio_b += items[b].ratios[o];
                }
                return ratio_a < ratio_b; // Crescente: menores ratios ficam no início
            });

            // Remove itens até caber na mochila
            for (int id : items_in_bag) {
                if (ind.total_weight <= max_capacity) break;
                
                ind.chromosome[id] = false; // Remove item
                ind.total_weight -= items[id].weight;
                for (int o = 0; o < NUM_OBJECTIVES; ++o) {
                    ind.fitness[o] -= items[id].profits[o];
                }
            }
        }
        ind.valid = true;
    }

    void init_population() {
        population.clear();
        for (int i = 0; i < pop_size; ++i) {
            Individual ind(num_items);
            for (int j = 0; j < num_items; ++j) {
                if (uniform_real_distribution<>(0.0, 1.0)(rng) < 0.5) {
                    ind.chromosome[j] = true;
                }
            }
            evaluate_and_repair(ind);
            population.push_back(ind);
        }
    }

    // Operador de seleção 1: Torneio Binário (Tournament Selection)
    const Individual& tournament_selection(int objective_idx) {
        int k = 2; 
        int best_idx = uniform_int_distribution<>(0, pop_size - 1)(rng);
        
        for (int i = 1; i < k; ++i) {
            int challenger = uniform_int_distribution<>(0, pop_size - 1)(rng);
            // Compara fitness APENAS no objetivo da sub-população atual
            if (population[challenger].fitness[objective_idx] > population[best_idx].fitness[objective_idx]) {
                best_idx = challenger;
            }
        }
        return population[best_idx];
    }

    // Operador de seleção 2: Roleta (Roulette Wheel Selection)
    const Individual& roulette_selection(int objective_idx) {
        double total_fit = 0;
        for (const auto& ind : population) total_fit += ind.fitness[objective_idx];

        double spin = uniform_real_distribution<>(0.0, total_fit)(rng);
        double current = 0;
        
        for (const auto& ind : population) {
            current += ind.fitness[objective_idx];
            if (current >= spin) return ind;
        }
        return population.back(); 
    }


    // Operador de Cruzamento: Ponto Único (One-Point Crossover)
    pair<Individual, Individual> crossover(const Individual& p1, const Individual& p2) {
        Individual c1(num_items), c2(num_items);
        int point = uniform_int_distribution<>(1, num_items - 1)(rng);

        for (int i = 0; i < num_items; ++i) {
            if (i < point) {
                c1.chromosome[i] = p1.chromosome[i];
                c2.chromosome[i] = p2.chromosome[i];
            } else {
                c1.chromosome[i] = p2.chromosome[i];
                c2.chromosome[i] = p1.chromosome[i];
            }
        }
        return {c1, c2};
    }

    // Operador de Mutação: Bit-Flip (Inversão de Bit)
    void mutate(Individual& ind) {
        for (int i = 0; i < num_items; ++i) {
            if (uniform_real_distribution<>(0.0, 1.0)(rng) < mutation_rate) {
                ind.chromosome[i] = !ind.chromosome[i]; // Inverte: 0->1 ou 1->0
            }
        }
    }

    // --- EXECUÇÃO PRINCIPAL (EVOLUÇÃO) ---
    vector<Individual> run(int generations) {
        init_population();
        int subpop_size = pop_size / SUBPOP_COUNT; 

        for (int g = 0; g < generations; ++g) {
            
            // --- LOG DE EVOLUÇÃO (Fitness Total das 3 Mochilas) ---
            if (log_evolution) {
                double best_total_fit = 0;
                double sum_total_fit_pop = 0;

                for(const auto& ind : population) {
                    // Calcula a SOMA dos 3 objetivos para este indivíduo
                    double current_ind_total = 0;
                    for(double f : ind.fitness) current_ind_total += f;

                    // Verifica se é o melhor da população (em termos de soma)
                    if (current_ind_total > best_total_fit) best_total_fit = current_ind_total;
                    
                    sum_total_fit_pop += current_ind_total;
                }
                
                double avg_total_fit = sum_total_fit_pop / pop_size;
                
                *log_evolution << current_size << "," << current_sel_name << "," 
                               << current_run_id << "," << g << "," 
                               << best_total_fit << "," << avg_total_fit << "\n";
            }
            // --------------------------------------------------

            vector<Individual> new_pop;
            
            // ESTRATÉGIA AEMMT: Sub-populações focadas
            // O loop itera sobre os objetivos (0, 1, 2).
            // A cada iteração, gera filhos otimizados para AQUELE objetivo específico.
            for (int obj = 0; obj < NUM_OBJECTIVES; ++obj) {
                int pairs_needed = subpop_size / 2;
                for (int i = 0; i < pairs_needed; ++i) {
                    Individual p1, p2;
                    
                    // Seleção de Pais (Focada no objetivo atual 'obj')
                    if (selection_method == 1) {
                        p1 = roulette_selection(obj);
                        p2 = roulette_selection(obj);
                    } else {
                        p1 = tournament_selection(obj);
                        p2 = tournament_selection(obj);
                    }

                    // Cruzamento
                    auto children = crossover(p1, p2);

                    // Mutação e Reparo
                    mutate(children.first);
                    evaluate_and_repair(children.first);
                    
                    mutate(children.second);
                    evaluate_and_repair(children.second);

                    new_pop.push_back(children.first);
                    new_pop.push_back(children.second);
                }
            }
            // Preenche o restante da população se a divisão não for exata
            while(new_pop.size() < pop_size) new_pop.push_back(population[0]); 
            
            population = new_pop;
        }
        return population;
    }
};

// ============================================================================
// MAIN - LOOP DE BENCHMARK
// ============================================================================
int main() {
    ofstream csv_pareto("fronteira_pareto_completa.csv");
    csv_pareto << "Size,Selection,Run,Obj1,Obj2,Obj3\n";

    ofstream csv_evo("evolucao_fitness.csv");
    csv_evo << "Size,Selection,Run,Generation,BestFit,AvgFit\n";

    vector<int> sizes = { 250, 500, 750, 1000 };
    vector<int> selections = { 1, 2 }; // 1=Roleta, 2=Torneio
    
    int total_runs = 30;    
    int generations = 300; 

    cout << "=== BENCHMARK C++: FIT TOTAL & FRONTEIRA ===" << endl;

    for (int size : sizes) {
        for (int sel : selections) {
            
            // Taxa de Mutação Dinâmica (1/N)
            double mutation_rate = 1.0 / (double)size;
            string sel_name = (sel == 1) ? "Roleta" : "Torneio";

            for (int run = 1; run <= total_runs; ++run) {
                cout << "\r[Proc] Size:" << setw(4) << size 
                          << " | Met:" << setw(8) << sel_name 
                          << " | Run:" << setw(2) << run << flush;

                AMMTSolver solver(size, sel, mutation_rate, 0.05);
                solver.set_logging(&csv_evo, size, sel_name, run);
                
                vector<Individual> final_pop = solver.run(generations);

                for (const auto& ind : final_pop) {
                    csv_pareto << size << "," << sel_name << "," << run << "," 
                        << ind.fitness[0] << "," 
                        << ind.fitness[1] << "," 
                        << ind.fitness[2] << "\n";
                }
            }
        }
    }

    cout << "\n\nConcluido! Pressione ENTER para sair..." << endl;
    cin.get();
    
    return 0;
}
