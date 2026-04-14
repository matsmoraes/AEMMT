#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <string>

using namespace std;

// ============================================================================
// GERADOR DE INSTÂNCIAS FIXAS
// Gera 20 instâncias por tamanho (250, 500, 750, 1000) com sementes fixas.
// Cada instância é salva em: instances/mokp_<size>_inst<id>.csv
// Formato CSV: weight, profit1, profit2, profit3, capacity (última linha)
// ============================================================================

const int NUM_OBJECTIVES = 3;
const double CAPACITY_RATIO = 0.5;
const int NUM_INSTANCES = 20;

struct Item {
    double weight;
    double profits[3];
};

void generate_instance(int size, int instance_id, unsigned long seed) {
    mt19937 rng(seed);
    uniform_real_distribution<> dist(10.0, 100.0);

    vector<Item> items(size);
    double total_weight = 0.0;

    for (int i = 0; i < size; ++i) {
        items[i].weight = dist(rng);
        total_weight += items[i].weight;
        for (int k = 0; k < NUM_OBJECTIVES; ++k) {
            items[i].profits[k] = dist(rng);
        }
    }

    double capacity = ceil(0.5 * total_weight);

    // Salva instância
    ostringstream fname;
    fname << "instances/mokp_" << size << "_inst" << setw(2) << setfill('0') << instance_id << ".csv";

    ofstream f(fname.str());
    f << fixed << setprecision(4);
    f << "weight,profit1,profit2,profit3\n";
    for (int i = 0; i < size; ++i) {
        f << items[i].weight << ","
          << items[i].profits[0] << ","
          << items[i].profits[1] << ","
          << items[i].profits[2] << "\n";
    }
    // Última linha: capacidade
    f << "CAPACITY," << capacity << ",0,0\n";
    f.close();

    cout << "  Gerada: " << fname.str() << " (seed=" << seed << ", cap=" << (int)capacity << ")\n";
}

int main() {
    system("mkdir -p instances");

    vector<int> sizes = {250, 500, 750, 1000};

    // Sementes fixas e reproduzíveis: combinação determinística de size e instance_id
    cout << "=== GERADOR DE INSTÂNCIAS FIXAS (MOKP) ===\n\n";

    for (int size : sizes) {
        cout << "[Tamanho " << size << " itens]\n";
        for (int inst = 1; inst <= NUM_INSTANCES; ++inst) {
            // Semente determinística: evita colisão entre tamanhos
            unsigned long seed = (unsigned long)size * 1000000UL + (unsigned long)inst * 12345UL + 987654321UL;
            generate_instance(size, inst, seed);
        }
        cout << "\n";
    }

    cout << "Concluído! " << (sizes.size() * NUM_INSTANCES) << " instâncias geradas em ./instances/\n";
    return 0;
}