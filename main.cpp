#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <string>
#include <chrono> // Importante para semente aleatória real

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

// Representa um Item disponível para colocar na mochila
struct Item {
    int id;
    double weight;
    vector<double> profits; // Lucro para obj 1, 2, 3...
    vector<double> ratios;  // Lucro/Peso para cada objetivo (usado no Repair)
};

// Representa um Indivíduo (Solução)
struct Individual {
    vector<bool> chromosome; // Vetor binário (true = pegou item)
    vector<double> fitness;  // Fitness para cada objetivo
    double total_weight;
    bool valid;

    // Construtor
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
// CLASSE DO SOLUCIONADOR
// ============================================================================
class AMMTSolver {
private:
    int num_items;
    int pop_size;
    int selection_method; // 1 = Roleta, 2 = Torneio
    double mutation_rate;
    double elitism_rate;
    double max_capacity;

    vector<Item> items;
    vector<Individual> population;
    
    // Gerador de números aleatórios moderno (Mersenne Twister)
    mt19937 rng;

public:
    // Construtor: Configura o problema
    AMMTSolver(int items_n, int sel_method, double mut_rate, double elit_rate) 
        : num_items(items_n), selection_method(sel_method), 
          mutation_rate(mut_rate), elitism_rate(elit_rate) {
        
        // Inicializa semente aleatória baseada no relógio do sistema
        // Isso garante que cada execução seja diferente
        unsigned seed = chrono::system_clock::now().time_since_epoch().count();
        rng.seed(seed);

        // Tamanho da população fixo (igual ao artigo)
        pop_size = 92; 
        
        // Gera os dados do problema assim que a classe é criada
        generate_instance();
    }

    // 1. Gera Instância Aleatória (Simula os ficheiros de entrada)
    void generate_instance() {
        items.clear();
        double total_weight_all = 0;

        for (int i = 0; i < num_items; ++i) {
            Item it;
            it.id = i;
            // Peso aleatório entre 10 e 100 (conforme artigo ZT1999)
            it.weight = uniform_real_distribution<>(10.0, 100.0)(rng);
            total_weight_all += it.weight;

            // Lucros aleatórios para os 3 objetivos
            for (int o = 0; o < NUM_OBJECTIVES; ++o) {
                double profit = uniform_real_distribution<>(10.0, 100.0)(rng);
                it.profits.push_back(profit);
                // Pré-calcula razão (Lucro/Peso) para o Greedy Repair
                it.ratios.push_back(profit / it.weight);
            }
            items.push_back(it);
        }
        // Capacidade é 50% do peso total
        max_capacity = total_weight_all * CAPACITY_RATIO;
    }

    // 2. Avalia e Repara (Greedy Repair) - O CORAÇÃO DO ALGORITMO
    void evaluate_and_repair(Individual& ind) {
        ind.total_weight = 0;
        fill(ind.fitness.begin(), ind.fitness.end(), 0.0);

        // Calcula peso e fitness inicial
        for (int i = 0; i < num_items; ++i) {
            if (ind.chromosome[i]) {
                ind.total_weight += items[i].weight;
                for (int o = 0; o < NUM_OBJECTIVES; ++o) {
                    ind.fitness[o] += items[i].profits[o];
                }
            }
        }

        // Se excedeu a capacidade, remove itens ruins
        if (ind.total_weight > max_capacity) {
            // Lista índices dos itens que estão na mochila
            vector<int> items_in_bag;
            for (int i = 0; i < num_items; ++i) {
                if (ind.chromosome[i]) items_in_bag.push_back(i);
            }

            // Ordena pelo "Pior Ratio Global" (Média dos ratios)
            // Itens com menor lucro/peso ficam no início para serem removidos
            sort(items_in_bag.begin(), items_in_bag.end(), [&](int a, int b) {
                double ratio_a = 0, ratio_b = 0;
                for(int o=0; o<NUM_OBJECTIVES; ++o) {
                    ratio_a += items[a].ratios[o];
                    ratio_b += items[b].ratios[o];
                }
                return ratio_a < ratio_b; 
            });

            // Remove itens até caber
            for (int id : items_in_bag) {
                if (ind.total_weight <= max_capacity) break;
                
                ind.chromosome[id] = false; // Remove
                ind.total_weight -= items[id].weight;
                for (int o = 0; o < NUM_OBJECTIVES; ++o) {
                    ind.fitness[o] -= items[id].profits[o];
                }
            }
        }
        ind.valid = true;
    }

    // 3. Inicializa População
    void init_population() {
        population.clear();
        for (int i = 0; i < pop_size; ++i) {
            Individual ind(num_items);
            for (int j = 0; j < num_items; ++j) {
                // 50% de chance de pegar cada item inicialmente
                if (uniform_real_distribution<>(0.0, 1.0)(rng) < 0.5) {
                    ind.chromosome[j] = true;
                }
            }
            evaluate_and_repair(ind);
            population.push_back(ind);
        }
    }

    // 4. Seleção por Torneio Binário
    const Individual& tournament_selection(int objective_idx) {
        int k = 2; 
        int best_idx = uniform_int_distribution<>(0, pop_size - 1)(rng);
        
        for (int i = 1; i < k; ++i) {
            int challenger = uniform_int_distribution<>(0, pop_size - 1)(rng);
            // Compara baseado APENAS no objetivo da sub-população atual
            if (population[challenger].fitness[objective_idx] > population[best_idx].fitness[objective_idx]) {
                best_idx = challenger;
            }
        }
        return population[best_idx];
    }

    // 5. Seleção por Roleta
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

    // 6. Crossover (Ponto Único)
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

    // 7. Mutação (Bit Flip)
    void mutate(Individual& ind) {
        for (int i = 0; i < num_items; ++i) {
            if (uniform_real_distribution<>(0.0, 1.0)(rng) < mutation_rate) {
                ind.chromosome[i] = !ind.chromosome[i]; 
            }
        }
    }

    // 8. Execução Principal (Evolução)
    double run(int generations) {
        init_population();
        
        int subpop_size = pop_size / SUBPOP_COUNT; 
        // Nota: Assumimos pop_size divisível por 3 (90, 93, etc.)

        for (int g = 0; g < generations; ++g) {
            vector<Individual> new_pop;
            
            // Loop para gerar nova população
            // Dividimos a geração em 3 partes, cada uma focando num objetivo (AMMT)
            for (int obj = 0; obj < NUM_OBJECTIVES; ++obj) {
                // Preenchemos 1/3 da nova população focando neste objetivo
                // Geramos 2 filhos por vez, então o loop roda metade das vezes necessárias
                int pairs_needed = subpop_size / 2;
                
                for (int i = 0; i < pairs_needed; ++i) {
                    Individual p1, p2;
                    
                    // Seleção
                    if (selection_method == 1) {
                        p1 = roulette_selection(obj);
                        p2 = roulette_selection(obj);
                    } else {
                        p1 = tournament_selection(obj);
                        p2 = tournament_selection(obj);
                    }

                    // Crossover
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
            
            // Preenche o restante se a divisão não for exata (opcional)
            while(new_pop.size() < pop_size) {
                new_pop.push_back(population[0]); 
            }

            population = new_pop;
        }

        // Retorna o melhor fitness do Objetivo 1 apenas como referência rápida
        // O ideal é salvar a Fronteira de Pareto num ficheiro
        double best_val = 0;
        for (const auto& ind : population) {
            if (ind.fitness[0] > best_val) best_val = ind.fitness[0];
        }
        return best_val;
    }
};

// ============================================================================
// RODANDO TUDO
// ============================================================================
int main() {
    // Ficheiro de saída
    ofstream csv("resultados_cpp.csv");
    csv << "Size,Selection,Mutation,Run,BestObjective1\n";

    // Configurações do Benchmark
    vector<int> sizes = { 250, 500, 750, 1000 };
    vector<int> selections = { 1, 2 }; // 1=Roleta, 2=Torneio
    
    // Configurações de execução
    int total_runs = 30;    // Quantidade de execuções para média estatística
    int generations = 300;  // Gerações por execução

    cout << "========================================" << endl;
    cout << "   BENCHMARK C++ MOKP (AMMT)           " << endl;
    cout << "========================================" << endl;

    for (int size : sizes) {
        for (int sel : selections) {
            
            // Taxa de mutação dinâmica (1 / N)
            double mutation_rate = 1.0 / (double)size;
            string sel_name = (sel == 1) ? "Roleta" : "Torneio";

            for (int run = 1; run <= total_runs; ++run) {
                
                // Exibe progresso no terminal
                cout << "\r[Processando] Itens: " << setw(4) << size 
                          << " | Metodo: " << setw(8) << sel_name 
                          << " | Run: " << setw(2) << run << flush;

                // Cria o Solver e roda
                AMMTSolver solver(size, sel, mutation_rate, 0.05);
                double result = solver.run(generations);

                // Salva no CSV
                csv << size << "," << sel_name << "," << mutation_rate << "," 
                    << run << "," << result << "\n";
            }
        }
    }

    cout << "\n\n========================================" << endl;
    cout << " CONCLUIDO! Resultados em 'resultados_cpp.csv'" << endl;
    cout << "========================================" << endl;

    
    cout << "Pressione ENTER para sair..." << endl;
    cin.get();
    
    return 0;
}
