# AMMT: Multi-Objective Knapsack Problem Solver

![C++](https://img.shields.io/badge/C++-17-blue.svg?style=flat&logo=c%2B%2B)
![Python](https://img.shields.io/badge/Python-3.8+-yellow.svg?style=flat&logo=python)
![Status](https://img.shields.io/badge/Status-Research-green)

## Project Overview

This project implements a high-performance **Evolutionary Algorithm (AMMT)** in **C++** designed to solve the **Multi-Objective Knapsack Problem (MOKP)**.

In combinatorial optimization, the MOKP is a classic problem where items have a specific weight and multiple conflicting profit values. The goal is to select a subset of items to maximize multiple objectives simultaneously without exceeding the capacity of the knapsack.

This repository contains:
1.  **C++ Solver:** A genetic algorithm engine optimized for performance, capable of handling large instances (up to 1000 items).
2.  **Python Analytics Pipeline:** Scripts to visualize convergence and calculate the **Hypervolume** metric using `pymoo`, comparing the results against state-of-the-art algorithms like **NSGA-III**.

## Directory Structure

```text
matsmoraes-aemmt/
├── main.cpp                # Core C++ source code (AMMT Solver)
├── plot_convergence.py     # Visualization of fitness evolution over generations
├── plot_final_cpp.py       # Hypervolume calculation and comparison (AMMT vs NSGA-III)
├── plot_preliminar_cpp.py  # Preliminary analysis of selection methods
└── resultados_cpp.csv      # Sample dataset from preliminary runs
```

## Genetic Operators & Strategy

The AEMMT algorithm distinguishes itself through a specific set of evolutionary operators tailored for the discrete nature of the Knapsack Problem.

### Population Strategy: Adaptive Sub-Populations

Instead of a single monolithic population, the AEMMT divides the total population (N=90) into sub-populations, each dedicated to optimizing a specific objective (Knapsack 1, Knapsack 2, Knapsack 3). This promotes specialized search while maintaining global diversity through migration or shared genetic material during crossover.

### Selection Operators

The algorithm implements and compares two distinct selection pressures:

* **Roulette Wheel Selection:** Probabilistic selection proportional to fitness. Preserves diversity in early stages but may suffer from slow convergence in large-scale instances.
* **Binary Tournament (k=2):** Two individuals are randomly picked, and the fitter one is selected. This method proved superior for high-dimensional problems (1000 items) by applying consistent selective pressure.

### Crossover: One-Point

* **Type:** One-Point Crossover.
* **Mechanism:** A random cut-point is selected in the chromosome vector. The parents swap segments to produce two offspring.
* **Rate:** 100% (Occurs for every selected pair).
* **Why:** Simple and effective for preserving building blocks in binary representations.

### Mutation: Bit-Flip

* **Type:** Bit-Flip Mutation.
* **Mechanism:** Each gene (item inclusion bit) is independently inverted (0→1 or 1→0) with probability $P_m$.
* **Rate:** Dynamic ($P_m = 1 / N$, where $N$ is the number of items).
* **Why:** Ensures that, on average, exactly one gene is mutated per individual, regardless of problem size (250 vs 1000 items), maintaining stable exploration without disrupting good solutions.

### Constraint Handling: Greedy Repair

A critical component of this solver is its ability to handle constraints efficiently. Most random solutions in MOKP are invalid (weight > capacity).

* **Method:** Greedy Repair Heuristic.
* **Process:**
    1. Identify if the knapsack is overweight.
    2. Calculate the **Profit/Weight Ratio** for all selected items.
    3. Iteratively **remove** items with the *lowest* efficiency ratio until the weight constraint is met.
* **Benefit:** This transforms infeasible solutions into feasible ones on the boundary of the feasible region, significantly accelerating convergence.

## Prerequisites & Requirements

To replicate the experiments and run the visualization scripts, you need the following environment:

### 1. C++ Compiler

* **GCC/G++** (Supporting C++11 standard or higher).
* *Windows users:* MinGW or Visual Studio Build Tools.
* *Linux/Mac users:* Standard `build-essential`.

### 2. Python Environment

The analysis scripts rely on Python 3.x and specific data science libraries. You can install all dependencies via pip:

```bash
pip install pandas matplotlib seaborn numpy pymoo

```

* **pandas & numpy:** For data manipulation and vector operations.
* **matplotlib & seaborn:** For generating high-quality scientific plots.
* **pymoo:** A specific framework for Multi-Objective Optimization, used here to calculate the Hypervolume indicator.

## Key Concepts & Metrics

### The Repair Heuristic

A critical component of this solver (`evaluate_and_repair` function in `main.cpp`) is its ability to handle constraints. Instead of simply discarding invalid solutions (chromosomes that exceed knapsack weight), the algorithm applies a **Greedy Repair Heuristic**:

1. It identifies items in the knapsack.
2. It calculates a ratio (Profit/Weight) for the selected items.
3. It iteratively removes the items with the *worst* efficiency ratios until the weight constraint is met.
This approach significantly accelerates convergence by guiding the population toward feasible regions of the search space.

### Why Hypervolume?

In Multi-Objective Optimization, there is rarely a single "best" solution. Instead, we seek a set of trade-off solutions known as the **Pareto Front**.

To evaluate the quality of our approximation, we use the **Hypervolume (HV)** metric:

* **Definition:** HV measures the volume of the objective space that is "dominated" by the solution set relative to a reference point.
* **Significance:** It is one of the few unary metrics that captures both **Convergence** (how close we are to the true optimum) and **Diversity** (how well spread our solutions are across the objectives).
* **In this project:** The `plot_final_cpp.py` script calculates the normalized HV to prove that the AMMT algorithm produces results competitive with (or superior to) NSGA-III.

## How to Run

### Step 1: Compile and Run the C++ Solver

First, compile the core logic to generate the raw data (`fronteira_pareto_completa.csv` and `evolucao_fitness.csv`).

```bash
# Compile with optimization flag -O3
g++ main.cpp -o benchmark_app -O3

# Run the executable
./benchmark_app

```

### Step 2: Visualizing Convergence

Generate the plot that shows how the algorithm improves over generations.

```bash
python plot_convergence.py

```

*Output: `analise_convergencia.png*`

### Step 3: Comparative Analysis (Hypervolume)

Run the final comparison script to calculate metrics and compare against literature data.

```bash
python plot_final_cpp.py

```

*Output: `comparacao_final_hv_roleta.png*`

## Results

The output images generated by the pipeline provide two main insights:

1. **Convergence Analysis:** Demonstrates the stability of the genetic algorithm, showing a steady increase in fitness across generations for different problem sizes (250, 500, 750, 1000 items).
2. **Algorithm Comparison:** The bar charts compare the **Mean Hypervolume** of the AMMT approach against the **NSGA-III** (based on data extracted from scientific literature). This validates the efficiency of the implemented Selection and Crossover operators.

## References

1. Deb K, Jain H (2014) An Evolutionary Many-Objective Optimization Algorithm Using Reference-Point-Based Nondominated Sorting Approach, Part I: Solving Problems With Box Constraints. *IEEE Transactions on Evolutionary Computation* 18(4).
2. Ishibuchi H, Imada R, Setoguchi Y, Nojima Y (2016) Performance comparison of NSGA-II and NSGA-III on various many-objective test problems. In: *2016 IEEE Congress on Evolutionary Computation (CEC)*, 24-29 July 2016.
3. Wangsom P, Lavangnananda K (2018) Extreme Solutions NSGA-III (E-NSGA-III) for Multi-Objective Constrained Problems. In: *2018 10th International Conference on Knowledge and Smart Technology (KST)*, Chiang Mai, Thailand.
4. Zitzler E, Thiele L (1998) Multiobjective Evolutionary Algorithms: A Comparative Case Study and the Strength Pareto Approach. *IEEE Transactions on Evolutionary Computation* 2(4).
5. Deb K (2001) *Multi-Objective Optimization Using Evolutionary Algorithms*. John Wiley & Sons, Inc.

## Author

**Matheus de Moraes Neves**

* [LinkedIn](https://www.linkedin.com/in/matheus-neves-864aa01a8/)
* [GitHub](https://github.com/matsmoraes)

---

*This project was developed as part of a Scientific Initiation research on Combinatorial Optimization and Artificial Intelligence.*

```
