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
// Formato CSV: weight, profit1, profit2, profit3
//              CAPACITY,<valor>,0,0  (última linha)
//
// Segue a especificação de Zitzler (1999) e Wangsom (2019):
//   - Pesos e lucros são inteiros em [10, 100]
//   - Capacidade = ceil(0.5 * soma_dos_pesos) [eq. 4 de Wangsom]
// ============================================================================

const int NUM_OBJECTIVES = 3;
const int NUM_INSTANCES  = 20;

struct Item {
    int weight;
    int profits[NUM_OBJECTIVES];
};

void generate_instance(int size, int instance_id, unsigned long seed) {
    mt19937 rng(seed);
    uniform_int_distribution<int> dist(10, 100);

    vector<Item> items(size);
    long total_weight = 0;

    for (int i = 0; i < size; ++i) {
        items[i].weight = dist(rng);
        total_weight += items[i].weight;
        for (int k = 0; k < NUM_OBJECTIVES; ++k)
            items[i].profits[k] = dist(rng);
    }

    int capacity = (int)ceil(0.5 * total_weight);

    // Monta nome do arquivo
    ostringstream fname;
    fname << "instances/mokp_" << size
          << "_inst" << setw(2) << setfill('0') << instance_id << ".csv";

    ofstream f(fname.str());
    if (!f.is_open()) {
        cerr << "ERRO: nao foi possivel criar '" << fname.str() << "'\n";
        return;
    }

    // Cabeçalho
    f << "weight,profit1,profit2,profit3\n";

    // Itens
    for (int i = 0; i < size; ++i) {
        f << items[i].weight     << ","
          << items[i].profits[0] << ","
          << items[i].profits[1] << ","
          << items[i].profits[2] << "\n";
    }

    // Capacidade (última linha)
    f << "CAPACITY," << capacity << ",0,0\n";
    f.close();

    cout << "  Gerada: " << fname.str()
         << "  (seed=" << seed << ", cap=" << capacity << ")\n";
}

int main() {
    system("mkdir -p instances");

    const vector<int> sizes = {250, 500, 750, 1000};

    cout << "=== GERADOR DE INSTÂNCIAS FIXAS (MOKP) ===\n\n";

    for (int size : sizes) {
        cout << "[Tamanho " << size << " itens]\n";
        for (int inst = 1; inst <= NUM_INSTANCES; ++inst) {
            // Semente determinística e sem colisão entre tamanhos/instâncias
            unsigned long seed = (unsigned long)size * 1000000UL
                               + (unsigned long)inst * 12345UL
                               + 987654321UL;
            generate_instance(size, inst, seed);
        }
        cout << "\n";
    }

    cout << "Concluido! "
         << (sizes.size() * NUM_INSTANCES)
         << " instancias geradas em ./instances/\n";
    return 0;
}