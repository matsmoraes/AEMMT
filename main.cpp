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
const int NUM_OBJECTIVES = 3;
const int SUBPOP_COUNT = 3;        // Uma sub-população por objetivo
const double CAPACITY_RATIO = 0.5; // Mochila aguenta 50% do peso total

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
    vector<bool> chromosome; 
    vector<double> fitness;  
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

    // Ficheiro para log de evolução (opcional)
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
        pop_size = 92; 
        log_evolution = nullptr;
        generate_instance();
    }

    // Configura o log de evolução
    void set_logging(ofstream* log_file, int size, string sel, int run) {
        log_evolution = log_file;
        current_size = size;
        current_sel_name = sel;
        current_run_id = run;
    }

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
                it.ratios.push_back(profit / it.weight);
            }
            items.push_back(it);
        }
        max_capacity = total_weight_all * CAPACITY_RATIO;
    }

    void evaluate_and_repair(Individual& ind) {
        ind.total_weight = 0;
        fill(ind.fitness.begin(), ind.fitness.end(), 0.0);

        for (int i = 0; i < num_items; ++i) {
            if (ind.chromosome[i]) {
                ind.total_weight += items[i].weight;
                for (int o = 0; o < NUM_OBJECTIVES; ++o) {
                    ind.fitness[o] += items[i].profits[o];
                }
            }
        }

        if (ind.total_weight > max_capacity) {
            vector<int> items_in_bag;
            for (int i = 0; i < num_items; ++i) {
                if (ind.chromosome[i]) items_in_bag.push_back(i);
            }
            sort(items_in_bag.begin(), items_in_bag.end(), [&](int a, int b) {
                double ratio_a = 0, ratio_b = 0;
                for(int o=0; o<NUM_OBJECTIVES; ++o) {
                    ratio_a += items[a].ratios[o];
                    ratio_b += items[b].ratios[o];
                }
                return ratio_a < ratio_b; 
            });

            for (int id : items_in_bag) {
                if (ind.total_weight <= max_capacity) break;
                ind.chromosome[id] = false;
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

    const Individual& tournament_selection(int objective_idx) {
        int k = 2; 
        int best_idx = uniform_int_distribution<>(0, pop_size - 1)(rng);
        for (int i = 1; i < k; ++i) {
            int challenger = uniform_int_distribution<>(0, pop_size - 1)(rng);
            if (population[challenger].fitness[objective_idx] > population[best_idx].fitness[objective_idx]) {
                best_idx = challenger;
            }
        }
        return population[best_idx];
    }

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

    void mutate(Individual& ind) {
        for (int i = 0; i < num_items; ++i) {
            if (uniform_real_distribution<>(0.0, 1.0)(rng) < mutation_rate) {
                ind.chromosome[i] = !ind.chromosome[i]; 
            }
        }
    }

    // --- EXECUÇÃO: Retorna TODA a população final ---
    vector<Individual> run(int generations) {
        init_population();
        int subpop_size = pop_size / SUBPOP_COUNT; 

        for (int g = 0; g < generations; ++g) {
            
            // --- LOG DE EVOLUÇÃO ---
            if (log_evolution) {
                double best_obj1 = 0;
                double sum_obj1 = 0;
                for(const auto& ind : population) {
                    if (ind.fitness[0] > best_obj1) best_obj1 = ind.fitness[0];
                    sum_obj1 += ind.fitness[0];
                }
                double avg_obj1 = sum_obj1 / pop_size;
                
                // Formato: Size, Selection, Run, Generation, BestFitness, AvgFitness
                *log_evolution << current_size << "," << current_sel_name << "," 
                               << current_run_id << "," << g << "," 
                               << best_obj1 << "," << avg_obj1 << "\n";
            }
            // -----------------------

            vector<Individual> new_pop;
            for (int obj = 0; obj < NUM_OBJECTIVES; ++obj) {
                int pairs_needed = subpop_size / 2;
                for (int i = 0; i < pairs_needed; ++i) {
                    Individual p1, p2;
                    if (selection_method == 1) {
                        p1 = roulette_selection(obj);
                        p2 = roulette_selection(obj);
                    } else {
                        p1 = tournament_selection(obj);
                        p2 = tournament_selection(obj);
                    }
                    auto children = crossover(p1, p2);
                    mutate(children.first);
                    evaluate_and_repair(children.first);
                    mutate(children.second);
                    evaluate_and_repair(children.second);
                    new_pop.push_back(children.first);
                    new_pop.push_back(children.second);
                }
            }
            while(new_pop.size() < pop_size) new_pop.push_back(population[0]); 
            population = new_pop;
        }
        return population;
    }
};

// ============================================================================
// MAIN - GERA FRONTEIRA E LOG DE EVOLUÇÃO
// ============================================================================
int main() {
    // 1. Arquivo para Fronteira de Pareto (Para Hypervolume)
    ofstream csv_pareto("fronteira_pareto_completa.csv");
    csv_pareto << "Size,Selection,Run,Obj1,Obj2,Obj3\n";

    // 2. Arquivo para Evolução do Fitness
    ofstream csv_evo("evolucao_fitness.csv");
    csv_evo << "Size,Selection,Run,Generation,BestFit,AvgFit\n";

    // Configurações
    vector<int> sizes = { 250, 500, 750, 1000 };
    vector<int> selections = { 1, 2 }; // 1=Roleta, 2=Torneio
    
    int total_runs = 30;    
    int generations = 300; 

    cout << "=== BENCHMARK C++: FRONTEIRA + CONVERGÊNCIA ===" << endl;

    for (int size : sizes) {
        for (int sel : selections) {
            double mutation_rate = 1.0 / (double)size;
            string sel_name = (sel == 1) ? "Roleta" : "Torneio";

            for (int run = 1; run <= total_runs; ++run) {
                cout << "\r[Proc] Size:" << setw(4) << size 
                          << " | Met:" << setw(8) << sel_name 
                          << " | Run:" << setw(2) << run << flush;

                AMMTSolver solver(size, sel, mutation_rate, 0.05);
                
                // Configura o log de evolução para salvar neste arquivo
                solver.set_logging(&csv_evo, size, sel_name, run);
                
                // Roda e pega população final
                vector<Individual> final_pop = solver.run(generations);

                // Salva Fronteira final
                for (const auto& ind : final_pop) {
                    csv_pareto << size << "," << sel_name << "," << run << "," 
                        << ind.fitness[0] << "," 
                        << ind.fitness[1] << "," 
                        << ind.fitness[2] << "\n";
                }
            }
        }
    }

    cout << "\n\nConcluido! Arquivos gerados:" << endl;
    cout << "1. fronteira_pareto_completa.csv (Para HV)" << endl;
    cout << "2. evolucao_fitness.csv (Para Convergencia)" << endl;
    
    cout << "Pressione ENTER para sair..." << endl;
    cin.get();
    
    return 0;
}