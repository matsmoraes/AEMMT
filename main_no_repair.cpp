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
#include <sstream>

using namespace std;

// ============================================================================
// CONFIGURACOES GLOBAIS
// ============================================================================
const int NUM_OBJECTIVES  = 3;
const int SUBPOP_COUNT    = 3;
const int ELITE_PER_SUBPOP = 1;

// ============================================================================
// ESTRUTURAS DE DADOS
// ============================================================================

struct Item {
    int    id;
    double weight;
    double profits[NUM_OBJECTIVES];
};

struct Individual {
    vector<bool> chromosome;
    double       fitness[NUM_OBJECTIVES];
    double       total_weight;
    bool         feasible; // true se nao viola restricao de capacidade

    Individual() : total_weight(0.0), feasible(false) {
        fill(fitness, fitness + NUM_OBJECTIVES, 0.0);
    }
    Individual(int n) : total_weight(0.0), feasible(false) {
        chromosome.resize(n, false);
        fill(fitness, fitness + NUM_OBJECTIVES, 0.0);
    }

    // Soma dos objetivos
    // Individuos inviaveis tem soma 0, nunca sao elite
    double total_fitness() const {
        if (!feasible) return 0.0;
        double s = 0;
        for (int o = 0; o < NUM_OBJECTIVES; ++o) s += fitness[o];
        return s;
    }
};

// ============================================================================
// CARREGAMENTO DE INSTANCIA
// ============================================================================
bool load_instance(const string& path, vector<Item>& items, double& capacity) {
    ifstream f(path);
    if (!f.is_open()) { cerr << "ERRO: nao foi possivel abrir '" << path << "'\n"; return false; }

    items.clear();
    string line;
    getline(f, line); // cabecalho

    int id = 0;
    while (getline(f, line)) {
        if (line.empty()) continue;
        if (line.substr(0, 8) == "CAPACITY") {
            capacity = stod(line.substr(line.find(',') + 1));
            break;
        }
        Item it; it.id = id++;
        istringstream ss(line); string tok;
        getline(ss, tok, ','); it.weight = stod(tok);
        for (int k = 0; k < NUM_OBJECTIVES; ++k) {
            getline(ss, tok, ',');
            it.profits[k] = stod(tok);
        }
        items.push_back(it);
    }
    return true;
}

// ============================================================================
// AEMMT
// ============================================================================
class AMMTSolver {
private:
    int            num_items;
    int            pop_size;
    int            selection_method;
    double         mutation_rate;
    vector<Item>   items;
    double         max_capacity;
    vector<Individual> population;
    mt19937        rng;

    ofstream* log_evolution;
    int    current_run_id, current_size, current_inst;
    string current_sel_name;

public:
    AMMTSolver(const vector<Item>& inst_items, double capacity,
               int sel_method, double mut_rate)
        : items(inst_items), max_capacity(capacity),
          num_items((int)inst_items.size()),
          selection_method(sel_method), mutation_rate(mut_rate)
    {
        rng.seed(chrono::system_clock::now().time_since_epoch().count());
        pop_size = 90;
        log_evolution = nullptr;
        current_run_id = current_size = current_inst = 0;
    }

    void set_logging(ofstream* f, int size, int inst, const string& sel, int run) {
        log_evolution = f; current_size = size; current_inst = inst;
        current_sel_name = sel; current_run_id = run;
    }

    // ------------------------------------------------------------------
    // Solucao inviavel recebe fitness = 0 em todos os objetivos
    // ------------------------------------------------------------------
    void evaluate(Individual& ind) {
        ind.total_weight = 0;
        fill(ind.fitness, ind.fitness + NUM_OBJECTIVES, 0.0);

        for (int i = 0; i < num_items; ++i) {
            if (ind.chromosome[i]) {
                ind.total_weight += items[i].weight;
                for (int o = 0; o < NUM_OBJECTIVES; ++o)
                    ind.fitness[o] += items[i].profits[o];
            }
        }

        if (ind.total_weight > max_capacity) {
            // marca como inviavel
            fill(ind.fitness, ind.fitness + NUM_OBJECTIVES, 0.0);
            ind.feasible = false;
        } else {
            ind.feasible = true;
        }
    }

    // ------------------------------------------------------------------
    // Inicializacao
    // ------------------------------------------------------------------
    void init_population() {
        population.clear();
        population.reserve(pop_size);
        for (int i = 0; i < pop_size; ++i) {
            Individual ind(num_items);
            for (int j = 0; j < num_items; ++j)
                ind.chromosome[j] = uniform_real_distribution<>(0.0,1.0)(rng) < 0.5;
            evaluate(ind);
            population.push_back(move(ind));
        }
    }

    // ------------------------------------------------------------------
    // Selecao: Torneio Binario
    // Inviavel perde sempre (fitness=0)
    // ------------------------------------------------------------------
    const Individual& tournament_selection(int obj) {
        int best = uniform_int_distribution<>(0, pop_size-1)(rng);
        int chal = uniform_int_distribution<>(0, pop_size-1)(rng);
        return population[population[chal].fitness[obj] >
                          population[best].fitness[obj] ? chal : best];
    }

    // ------------------------------------------------------------------
    // Selecao: Roleta
    // ------------------------------------------------------------------
    const Individual& roulette_selection(int obj) {
        double total = 0;
        for (const auto& ind : population) total += ind.fitness[obj];
        if (total == 0) return population[uniform_int_distribution<>(0, pop_size-1)(rng)];
        double spin = uniform_real_distribution<>(0.0, total)(rng);
        double cur  = 0;
        for (const auto& ind : population) {
            cur += ind.fitness[obj];
            if (cur >= spin) return ind;
        }
        return population.back();
    }

    // ------------------------------------------------------------------
    // Cruzamento: Ponto Unico
    // ------------------------------------------------------------------
    pair<Individual,Individual> crossover(const Individual& p1, const Individual& p2) {
        Individual c1(num_items), c2(num_items);
        int point = uniform_int_distribution<>(1, num_items-1)(rng);
        for (int i = 0; i < num_items; ++i) {
            c1.chromosome[i] = (i < point) ? p1.chromosome[i] : p2.chromosome[i];
            c2.chromosome[i] = (i < point) ? p2.chromosome[i] : p1.chromosome[i];
        }
        return {c1, c2};
    }

    // ------------------------------------------------------------------
    // Mutacao: Bit-Flip
    // ------------------------------------------------------------------
    void mutate(Individual& ind) {
        for (int i = 0; i < num_items; ++i)
            if (uniform_real_distribution<>(0.0,1.0)(rng) < mutation_rate)
                ind.chromosome[i] = !ind.chromosome[i];
    }

    // ------------------------------------------------------------------
    // Elitismo: melhores individuos VIAVEIS por objetivo
    // ------------------------------------------------------------------
    vector<Individual> extract_elite() {
        vector<Individual> elite;
        elite.reserve(SUBPOP_COUNT * ELITE_PER_SUBPOP);
        for (int obj = 0; obj < SUBPOP_COUNT; ++obj) {
            vector<int> idx(pop_size);
            iota(idx.begin(), idx.end(), 0);
            // Ordena decrescente por objetivo — inviavel (fitness=0) cai naturalmente
            sort(idx.begin(), idx.end(), [&](int a, int b) {
                return population[a].fitness[obj] > population[b].fitness[obj];
            });
            for (int k = 0; k < ELITE_PER_SUBPOP; ++k)
                if (population[idx[k]].feasible)   // preserva viaveis
                    elite.push_back(population[idx[k]]);
        }
        return elite;
    }

    // ------------------------------------------------------------------
    // LOOP PRINCIPAL
    // ------------------------------------------------------------------
    vector<Individual> run(int generations) {
        init_population();
        int subpop_size = pop_size / SUBPOP_COUNT; // 30

        for (int g = 0; g < generations; ++g) {

            // Log
            if (log_evolution) {
                double best = 0, sum = 0;
                for (const auto& ind : population) {
                    double t = ind.total_fitness();
                    if (t > best) best = t;
                    sum += t;
                }
                *log_evolution
                    << current_size     << ","
                    << current_inst     << ","
                    << current_sel_name << ","
                    << current_run_id   << ","
                    << g                << ","
                    << best             << ","
                    << (sum / pop_size) << "\n";
            }

            vector<Individual> elite = extract_elite();

            vector<Individual> new_pop;
            new_pop.reserve(pop_size);

            for (int obj = 0; obj < NUM_OBJECTIVES; ++obj) {
                int pairs = subpop_size / 2;
                for (int i = 0; i < pairs; ++i) {
                    Individual p1, p2;
                    if (selection_method == 1) {
                        p1 = roulette_selection(obj);
                        p2 = roulette_selection(obj);
                    } else {
                        p1 = tournament_selection(obj);
                        p2 = tournament_selection(obj);
                    }
                    auto [c1, c2] = crossover(p1, p2);
                    mutate(c1); evaluate(c1);
                    mutate(c2); evaluate(c2);
                    new_pop.push_back(move(c1));
                    new_pop.push_back(move(c2));
                }
            }

            // Reintroduz elite substituindo os piores (inviavel ou menor fitness)
            sort(new_pop.begin(), new_pop.end(), [](const Individual& a, const Individual& b) {
                return a.total_fitness() < b.total_fitness();
            });
            for (int e = 0; e < (int)elite.size() && e < (int)new_pop.size(); ++e)
                new_pop[e] = elite[e];

            population = move(new_pop);
        }
        return population;
    }
};

// Retorna o caminho do CSV de uma instancia
string instance_path(int size, int inst_id) {
    ostringstream ss;
    ss << "instances/mokp_" << size
       << "_inst" << setw(2) << setfill('0') << inst_id << ".csv";
    return ss.str();
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
    ofstream csv_pareto("fronteira_pareto_completa.csv");
    csv_pareto << "Size,Instance,Selection,Run,Obj1,Obj2,Obj3\n";

    ofstream csv_evo("evolucao_fitness.csv");
    csv_evo << "Size,Instance,Selection,Run,Generation,BestFit,AvgFit\n";

    const vector<int> sizes      = {250, 500, 750, 1000};
    const vector<int> selections = {1, 2}; // 1=Roleta, 2=Torneio
    const int INSTANCES  = 20;
    const int RUNS       = 30;
    const int GENERATIONS = 300;

    cout << "=== BENCHMARK AEMMT SEM REPARO (Death Penalty) ===\n\n";

    int total = (int)sizes.size() * (int)selections.size() * INSTANCES * RUNS;
    int done  = 0;

    for (int size : sizes) {
        for (int sel : selections) {
            string sel_name    = (sel == 1) ? "Roleta" : "Torneio";
            double mut_rate    = 1.0 / (double)size;

            for (int inst = 1; inst <= INSTANCES; ++inst) {
                vector<Item> items;
                double capacity = 0;
                string fpath = instance_path(size, inst);

                if (!load_instance(fpath, items, capacity)) {
                    cerr << "Pulando: " << fpath << "\n";
                    continue;
                }

                for (int run = 1; run <= RUNS; ++run) {
                    ++done;
                    cout << "\r[" << setw(5) << done << "/" << total << "]"
                         << " Size:" << setw(4) << size
                         << " Inst:" << setw(2) << inst
                         << " Sel:"  << setw(8) << sel_name
                         << " Run:"  << setw(2) << run << flush;

                    AMMTSolver solver(items, capacity, sel, mut_rate);
                    solver.set_logging(&csv_evo, size, inst, sel_name, run);

                    vector<Individual> final_pop = solver.run(GENERATIONS);

                    // Salva apenas solucoes viaveis na fronteira
                    for (const auto& ind : final_pop) {
                        if (ind.feasible) {
                            csv_pareto << size     << ","
                                       << inst     << ","
                                       << sel_name << ","
                                       << run      << ","
                                       << ind.fitness[0] << ","
                                       << ind.fitness[1] << ","
                                       << ind.fitness[2] << "\n";
                        }
                    }
                }
            }
        }
    }

    cout << "\n\nConcluido! CSVs gerados:\n"
         << "  fronteira_pareto_completa.csv\n"
         << "  evolucao_fitness.csv\n\n"
         << "Pressione ENTER para sair...";
    cin.get();
    return 0;
}