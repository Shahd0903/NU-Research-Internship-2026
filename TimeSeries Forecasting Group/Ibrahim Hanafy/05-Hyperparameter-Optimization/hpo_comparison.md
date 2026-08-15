# How Hyperparameter Optimizers Work (and Which to Use)

## Table of Contents

1. [The Core Problem](#the-core-problem)
2. [How Optimization Works for Hyperparameters](#how-optimization-works-for-hyperparameters)
3. [Category 1: Sequential Model-Based (Bayesian)](#category-1-sequential-model-based-bayesian)
4. [Category 2: Metaheuristic / Population-Based](#category-2-metaheuristic--population-based)
5. [Category 3: Bandit-Based (Early Stopping)](#category-3-bandit-based-early-stopping)
6. [Deep Dive: Grey Wolf Optimizer (GWO)](#deep-dive-grey-wolf-optimizer-gwo)
7. [Deep Dive: MealPy — The Metaheuristic Library](#deep-dive-mealpy--the-metaheuristic-library)
8. [Head-to-Head Comparison](#head-to-head-comparison)
9. [Why TPE (Optuna) for Our TCN](#why-tpe-optuna-for-our-tcn)
10. [When Metaheuristics Win](#when-metaheuristics-win)
11. [Decision Flowchart](#decision-flowchart)

---

## The Core Problem

You have a neural network (TCN). It has **hyperparameters** — settings you choose before training:

```
n_blocks  = ?    (2, 3, 4, 5, or 6)
n_filters = ?    (32, 64, 128, or 256)
kernel    = ?    (2, 3, 5, or 7)
dropout   = ?    (0.05 to 0.40, continuous)
lr        = ?    (0.0001 to 0.01, continuous)
batch     = ?    (64, 128, 256, or 512)
```

Each combination = one **trial**. You train the model, get an RMSE. You want the combination that gives the **lowest RMSE**.

**The catch:** Each trial takes 5–30 minutes. You can't try all combinations — there are `5 × 4 × 4 × 8 × continuous × 4 = tens of thousands`. You need a strategy to find good combos **fast**.

This is a **black-box optimization** problem:
- **Black-box** = you don't have a formula for RMSE(hyperparameters). You can only evaluate it by training.
- **Expensive** = each evaluation costs minutes of GPU time.
- **Noisy** = same hyperparameters can give slightly different RMSE due to random initialization.
- **Mixed** = some params are integers, some categorical, some continuous.

---

## How Optimization Works for Hyperparameters

Every optimizer follows the same loop:

```
1. PROPOSE a set of hyperparameters
2. EVALUATE them (train model, get RMSE)
3. UPDATE your strategy based on the result
4. REPEAT until budget exhausted
```

**What differs is Step 1 and Step 3** — how proposals are generated and how results inform future proposals.

### The Landscape Analogy

Imagine RMSE as a mountainous landscape. Each hyperparameter is a dimension. You're trying to find the lowest valley (minimum RMSE) by walking around, but:
- You can't see the landscape (black-box)
- Each step takes 10 minutes (expensive)
- The ground is foggy and shaky (noisy)
- Some directions are discrete jumps, not smooth steps (categorical params)

Different optimizers navigate this differently.

---

## Category 1: Sequential Model-Based (Bayesian)

### Core Idea
**Build a cheap model (surrogate) of the expensive objective.** Use the surrogate to decide where to evaluate next.

### TPE (Tree-structured Parzen Estimator) — used by Optuna

**How it works, step by step:**

```
1. Run 5-10 random trials (startup)
2. Split all observed results into two groups:
   - "Good" group: top 25% of trials (low RMSE)
   - "Bad" group:  bottom 75%
3. For each hyperparameter, build a probability distribution:
   - l(x) = probability of x in the "good" group
   - g(x) = probability of x in the "bad" group
4. Pick next trial by maximizing l(x) / g(x)
   (= high probability of being good, low probability of being bad)
5. Evaluate, add result, go to step 2
```

**Example in action:**

```
Trial  1: blocks=4, filters=256, lr=0.005  → RMSE=0.042  (random)
Trial  2: blocks=2, filters=64,  lr=0.001  → RMSE=0.035  (random)
Trial  3: blocks=3, filters=128, lr=0.008  → RMSE=0.038  (random)
Trial  4: blocks=5, filters=32,  lr=0.002  → RMSE=0.040  (random)
Trial  5: blocks=3, filters=64,  lr=0.003  → RMSE=0.031  (random)
--- startup done, TPE kicks in ---
TPE sees: "good" trials have blocks~3, filters~64, lr~0.001-0.003
Trial  6: blocks=3, filters=64,  lr=0.002  → RMSE=0.029  ← exploits
Trial  7: blocks=4, filters=64,  lr=0.001  → RMSE=0.030  ← nearby
Trial  8: blocks=2, filters=128, lr=0.004  → RMSE=0.037  ← explores
Trial  9: blocks=3, filters=64,  lr=0.0015 → RMSE=0.028  ← refines
...converges
```

**Strengths:**
- Learns from every past trial — gets smarter over time
- Handles mixed param types (categorical, int, float) natively
- Sample efficient: finds good configs in 20–50 trials
- Pairs with pruning (kill bad trials early)

**Weaknesses:**
- Sequential by nature (each trial depends on all previous)
- Surrogate model can be wrong early (not enough data)
- May over-exploit (get stuck near a local minimum)

### Gaussian Process (GP) — used by scikit-optimize

Same Bayesian idea, different surrogate model:
- Uses a Gaussian Process to model RMSE as a smooth function
- Provides uncertainty estimates: "I'm confident here, unsure there"
- Uses an **acquisition function** (EI, UCB) to balance exploration vs exploitation

**Better than TPE for:** purely continuous spaces, very small budgets (<20 trials)
**Worse than TPE for:** categorical params, high dimensions (>10 params), many trials

---

## Category 2: Metaheuristic / Population-Based

### Core Idea
**Maintain a population of candidate solutions. Evolve them using nature-inspired rules.** No surrogate model — the population IS the memory.

### GWO (Grey Wolf Optimizer) — available in mealpy

**Inspiration:** Wolf pack hierarchy. Alpha (best), Beta (2nd), Delta (3rd), Omega (rest).

**How it works:**

```
1. Initialize population of N wolves (random hyperparameters)
   Wolf 1: [blocks=3, filters=128, lr=0.005, ...]  → RMSE=0.038
   Wolf 2: [blocks=5, filters=64,  lr=0.001, ...]  → RMSE=0.033
   Wolf 3: [blocks=2, filters=256, lr=0.008, ...]  → RMSE=0.045
   ...
   Wolf N: [blocks=4, filters=32,  lr=0.002, ...]  → RMSE=0.041

2. Rank by RMSE:
   Alpha (α) = Wolf 2 (RMSE=0.033)  ← best
   Beta  (β) = Wolf 1 (RMSE=0.038)  ← 2nd best
   Delta (δ) = Wolf N (RMSE=0.041)  ← 3rd best

3. Each omega wolf updates position toward α, β, δ:
   new_position = (pos_toward_α + pos_toward_β + pos_toward_δ) / 3
   + random perturbation that shrinks over iterations

4. Evaluate all wolves at new positions
5. Update α, β, δ ranks
6. Repeat for T iterations
```

**The key equation (simplified):**

```python
# For each wolf i, for each hyperparameter dimension d:
A = 2 * a * random() - a        # a decreases from 2→0 over iterations
C = 2 * random()

X1 = alpha_pos[d] - A * |C * alpha_pos[d] - wolf_pos[d]|
X2 = beta_pos[d]  - A * |C * beta_pos[d]  - wolf_pos[d]|
X3 = delta_pos[d] - A * |C * delta_pos[d] - wolf_pos[d]|

wolf_pos[d] = (X1 + X2 + X3) / 3
```

Early iterations: `a` is large → big jumps (exploration)
Late iterations: `a` is small → fine adjustments (exploitation)

**Problem for HPO:**
- All positions are **continuous**. `filters=87.3` doesn't exist — must round to 64 or 128
- Each iteration evaluates **entire population**: N wolves × T iterations = N×T total evals
- No pruning — every wolf trains to completion

### PSO (Particle Swarm Optimization)

**Inspiration:** Bird flocking. Each particle remembers its personal best and follows the swarm's global best.

```python
velocity[d] = w * velocity[d]                         # inertia
            + c1 * random() * (personal_best[d] - pos[d])  # memory
            + c2 * random() * (global_best[d] - pos[d])    # social

position[d] = position[d] + velocity[d]
```

Similar strengths/weaknesses as GWO. Better at smooth landscapes, same categorical param problem.

### GA (Genetic Algorithm)

**Inspiration:** Natural selection. Crossover (combine two parents) + mutation (random change).

```
Parent A: [blocks=3, filters=64,  lr=0.001, dropout=0.1]  RMSE=0.029
Parent B: [blocks=4, filters=128, lr=0.003, dropout=0.2]  RMSE=0.031

Crossover:
Child 1:  [blocks=3, filters=128, lr=0.001, dropout=0.2]  ← mixed
Child 2:  [blocks=4, filters=64,  lr=0.003, dropout=0.1]  ← mixed

Mutation (5% chance per gene):
Child 1:  [blocks=3, filters=128, lr=0.001, dropout=0.15] ← dropout mutated
```

**Best metaheuristic for HPO** because crossover naturally handles mixed types. Still needs full population evaluation per generation.

---

## Category 3: Bandit-Based (Early Stopping)

### Hyperband / ASHA

**Core Idea:** Don't finish training bad configs. Start many, kill most early, promote survivors.

```
Bracket 0 (most aggressive):
  Rung 0: 81 configs × 1 epoch   → keep top 27
  Rung 1: 27 configs × 3 epochs  → keep top 9
  Rung 2:  9 configs × 9 epochs  → keep top 3
  Rung 3:  3 configs × 27 epochs → pick best

Bracket 1 (less aggressive):
  Rung 0: 27 configs × 3 epochs  → keep top 9
  Rung 1:  9 configs × 9 epochs  → keep top 3
  Rung 2:  3 configs × 27 epochs → pick best
```

**Assumption:** Early performance correlates with final performance. Usually true for neural networks, but not always (some configs are slow starters).

**Optuna combines TPE + Hyperband:** TPE picks configs intelligently, Hyperband kills bad ones early. Best of both worlds.

---

## Deep Dive: Grey Wolf Optimizer (GWO)

### Biological Inspiration

Grey wolves (*Canis lupus*) live in packs of 5–12 with a strict dominance hierarchy:

```
         ┌─────────┐
         │  Alpha α │  ← Makes all decisions (hunting, sleeping site, wake time)
         └────┬────┘
              │
         ┌────┴────┐
         │  Beta β  │  ← Second-in-command, enforces alpha's decisions
         └────┬────┘
              │
         ┌────┴────┐
         │ Delta δ  │  ← Subordinate to alpha/beta, dominates omega
         └────┬────┘
              │
    ┌─────────┴─────────┐
    │  Omega ω  ω  ω  ω │  ← Lowest rank, follows all others
    └───────────────────┘
```

In the wild, alpha leads the hunt. Beta and delta help encircle prey. Omegas follow. The pack converges on prey from multiple directions, gradually tightening the circle until they strike.

GWO (proposed by Mirjalili et al., 2014) translates this into math: **the three best solutions found so far guide all other solutions toward the optimum**.

### The Algorithm in Detail

#### Phase 1: Initialization

Generate N random candidate solutions ("wolves") in the search space:

```python
# Suppose we have 3 hyperparameters to optimize (simplified)
# Real ranges: n_blocks ∈ [2,6], lr ∈ [0.0001, 0.01], dropout ∈ [0.05, 0.4]

Wolf  1: [3.7, 0.0054, 0.22]  →  evaluate → RMSE = 0.038
Wolf  2: [5.1, 0.0012, 0.31]  →  evaluate → RMSE = 0.033
Wolf  3: [2.4, 0.0081, 0.08]  →  evaluate → RMSE = 0.045
Wolf  4: [4.8, 0.0023, 0.15]  →  evaluate → RMSE = 0.041
Wolf  5: [3.2, 0.0037, 0.28]  →  evaluate → RMSE = 0.036
...
Wolf 20: [4.1, 0.0019, 0.19]  →  evaluate → RMSE = 0.039
```

**Cost so far:** 20 evaluations — already 20 full model trainings.

#### Phase 2: Rank and Assign Roles

Sort by fitness (RMSE), assign hierarchy:

```
Alpha (α) = Wolf  2  →  RMSE = 0.033  (best)
Beta  (β) = Wolf  5  →  RMSE = 0.036  (2nd best)
Delta (δ) = Wolf  1  →  RMSE = 0.038  (3rd best)
Omega     = Everyone else
```

#### Phase 3: Encircling Prey (The Core Math)

Each omega wolf updates its position by averaging its attraction toward alpha, beta, and delta. This mimics wolves encircling prey from different directions.

**The coefficient vectors:**

```python
# 'a' is the convergence parameter — THE key control variable
# It linearly decreases from 2 to 0 over the total iterations T
a = 2 - (2 * current_iteration / max_iterations)

# For EACH wolf, for EACH dimension d:
# Generate random coefficients
r1, r2 = random(), random()     # uniform in [0, 1]
A = 2 * a * r1 - a              # A ∈ [-a, +a]
C = 2 * r2                      # C ∈ [0, 2]
```

**What A and C control:**

| Parameter | Range | Effect |
|-----------|-------|--------|
| `a = 2.0` (early) | A ∈ [-2, +2] | Large jumps — **exploration** |
| `a = 0.5` (late) | A ∈ [-0.5, +0.5] | Small adjustments — **exploitation** |
| `|A| > 1` | — | Wolf diverges from prey (explores new regions) |
| `|A| < 1` | — | Wolf converges toward prey (refines solution) |
| `C` | [0, 2] | Weights how much the leader's position matters. Random C prevents stagnation. |

**The position update equations:**

```python
# Distance between wolf and each leader:
D_alpha = abs(C1 * alpha_pos[d] - wolf_pos[d])
D_beta  = abs(C2 * beta_pos[d]  - wolf_pos[d])
D_delta = abs(C3 * delta_pos[d] - wolf_pos[d])

# Position pulled toward each leader:
X1 = alpha_pos[d] - A1 * D_alpha    # step toward alpha
X2 = beta_pos[d]  - A2 * D_beta     # step toward beta
X3 = delta_pos[d] - A3 * D_delta    # step toward delta

# Final position = average of all three influences:
wolf_pos[d] = (X1 + X2 + X3) / 3
```

**Why three leaders, not just alpha?** Using only alpha would cause all wolves to converge to one point (premature convergence). Beta and delta provide alternative directions, maintaining search diversity. The pack explores a triangular region defined by the three best solutions.

#### Phase 4: Iteration-by-Iteration Behavior

```
Iteration 1-10  (a: 2.0 → 1.3):
  |A| frequently > 1 → wolves jump far from leaders
  Pack EXPLORES broadly, covers large regions of search space
  Many wolves end up far from current best
  Diversity is high, convergence is low

Iteration 10-20 (a: 1.3 → 0.7):
  |A| sometimes > 1, sometimes < 1 → mixed behavior
  TRANSITION phase: exploration decreasing, exploitation increasing
  Pack starts forming clusters near promising regions

Iteration 20-30 (a: 0.7 → 0.0):
  |A| almost always < 1 → wolves converge tightly
  Pack EXPLOITS: all wolves close to alpha/beta/delta
  Fine-tuning: small position changes refine the best solution
  Diversity is low, convergence is high
```

#### Full Algorithm Pseudocode

```python
def grey_wolf_optimizer(objective_fn, bounds, n_wolves=20, max_iter=30):
    # Phase 1: Initialize
    wolves = random_positions(n_wolves, bounds)
    fitness = [objective_fn(w) for w in wolves]  # N evaluations
    
    # Phase 2: Initial ranking
    alpha, beta, delta = top_three(wolves, fitness)
    
    for t in range(max_iter):
        a = 2 - 2 * t / max_iter          # a: 2 → 0
        
        for i in range(n_wolves):
            for d in range(n_dimensions):
                # Generate random coefficients for each leader
                A1, C1 = 2*a*random()-a, 2*random()
                A2, C2 = 2*a*random()-a, 2*random()
                A3, C3 = 2*a*random()-a, 2*random()
                
                # Encircle: compute distance to each leader
                D1 = abs(C1 * alpha[d] - wolves[i][d])
                D2 = abs(C2 * beta[d]  - wolves[i][d])
                D3 = abs(C3 * delta[d] - wolves[i][d])
                
                # Update: average of three directed steps
                X1 = alpha[d] - A1 * D1
                X2 = beta[d]  - A2 * D2
                X3 = delta[d] - A3 * D3
                
                wolves[i][d] = (X1 + X2 + X3) / 3
            
            # Clip to bounds
            wolves[i] = clip(wolves[i], bounds)
            
            # Evaluate — THIS IS THE EXPENSIVE PART
            fitness[i] = objective_fn(wolves[i])   # Full model training!
        
        # Update hierarchy
        alpha, beta, delta = top_three(wolves, fitness)
    
    return alpha  # Best solution found
    
    # Total evaluations: n_wolves × max_iter = 20 × 30 = 600
```

### GWO Strengths (Why Researchers Like It)

1. **Few hyperparameters of its own** — only `n_wolves` and `max_iter`. Compare to GA (crossover rate, mutation rate, selection pressure, elitism count) or DE (F, CR, strategy).

2. **Smooth exploration→exploitation transition** — parameter `a` decreasing from 2→0 naturally balances exploration and exploitation. No abrupt switching or manual scheduling.

3. **Implicit diversity maintenance** — using three leaders (not one) prevents premature convergence. Wolves approach a triangular region, not a single point.

4. **No gradient needed** — works on any function: discontinuous, noisy, multimodal, black-box.

5. **Parallelizable** — all wolves evaluate independently within each iteration. With 20 GPUs, each iteration takes one evaluation time instead of 20.

### GWO Weaknesses (Why It Struggles for HPO)

1. **All operations are continuous** — position updates use arithmetic (subtraction, multiplication, averaging). `n_blocks = (3.2 + 4.7 + 2.1) / 3 = 3.33` — what does 3.33 blocks mean? Must round to 3, losing the algorithm's intended precision.

2. **No memory across iterations (except leaders)** — a wolf at position X that got RMSE=0.029 might move away next iteration and never return. Only alpha/beta/delta positions are preserved. TPE, by contrast, remembers EVERY past evaluation.

3. **Population waste** — iteration 15, wolf 7 might be at an obviously bad position, but GWO still evaluates it. No pruning, no early stopping. Every wolf trains to completion.

4. **Evaluation budget** — for HPO with 10-minute evaluations:
   ```
   20 wolves × 30 iterations = 600 evaluations
   600 × 10 minutes = 6,000 minutes = 100 hours
   
   vs. Optuna TPE:
   50 trials × 10 minutes = 500 minutes ≈ 8 hours
   (with pruning, even less — bad trials killed at epoch 5)
   ```

5. **Categorical parameters require encoding tricks:**
   ```
   filters ∈ {32, 64, 128, 256}
   
   Option A: Map to [0, 3], let GWO operate on continuous [0, 3], round to nearest int
   Problem: GWO treats 32→64 as same distance as 64→128, but 64→128 is 2× the filters
   
   Option B: Map to continuous and use log scale
   Problem: Still creates artifacts at boundaries
   
   Option C: One-hot encode
   Problem: Explodes dimensionality, wastes GWO's continuous arithmetic
   ```

### Worked Example: GWO for HPO vs. TPE

**Setup:** Optimize 3 hyperparameters, budget = 30 evaluations.

```
GWO: 10 wolves × 3 iterations = 30 evaluations
TPE: 30 sequential trials (10 random startup + 20 guided)
```

**GWO's 30 evaluations:**
```
Iteration 1 (random init, a=2.0):
  Wolf 1-10 evaluated at random positions → 10 evaluations
  Best found: RMSE = 0.033

Iteration 2 (a=1.33):
  All 10 wolves move toward top 3 with large perturbation
  Wolf 1-10 re-evaluated → 10 more evaluations (20 total)
  Some wolves found worse positions than iteration 1!
  Best found: RMSE = 0.031

Iteration 3 (a=0.67):
  Wolves converge closer to leaders
  Wolf 1-10 re-evaluated → 10 more evaluations (30 total)
  Best found: RMSE = 0.030
```

**TPE's 30 evaluations:**
```
Trials 1-10 (random):
  10 random configs → best RMSE = 0.033

Trials 11-20 (guided):
  TPE sees patterns: "low lr + medium filters → good RMSE"
  Focuses on promising region
  Each trial informed by ALL 10+ previous results
  Best found: RMSE = 0.028

Trials 21-30 (refined):
  TPE has 20 data points, surrogate model is accurate
  Exploits best region while occasionally exploring
  Best found: RMSE = 0.026
```

**Result with same budget:** TPE (0.026) beats GWO (0.030) because:
- TPE uses ALL 30 evaluations to build an increasingly accurate model
- GWO "forgets" positions — wolf at RMSE=0.031 in iteration 2 might move to RMSE=0.035 in iteration 3
- TPE's surrogate becomes more accurate with each trial; GWO only tracks three best positions

---

## Deep Dive: MealPy — The Metaheuristic Library

### What Is MealPy?

**MealPy** (Metaheuristic Algorithms in Python) is an open-source library that implements **200+ nature-inspired optimization algorithms** under a unified API. Created by Nguyen Van Thieu (first released 2020, actively maintained).

Think of it as the **scikit-learn of metaheuristics** — same interface for all algorithms, swap one optimizer for another by changing one line of code.

### Why MealPy Exists

Before MealPy, using metaheuristics meant:
- Finding individual implementations on GitHub (varying quality)
- Each with different APIs, input formats, output formats
- No standardized benchmarking
- Reimplementing algorithms from papers (error-prone)

MealPy solves this by providing:
- **Unified API** — every algorithm uses the same `solve()` interface
- **200+ algorithms** — from classics (GA, PSO) to obscure bio-inspired ones
- **Built-in transfer functions** — handles continuous→discrete conversion
- **Benchmarking tools** — compare algorithms on standard test functions

### Architecture Overview

```
                        ┌──────────────────────┐
                        │      Your Problem     │
                        │  (objective function  │
                        │   + bounds/types)     │
                        └──────────┬───────────┘
                                   │
                        ┌──────────▼───────────┐
                        │     Problem Class     │
                        │  bounds, obj_func,    │
                        │  minmax="min"         │
                        └──────────┬───────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │     Optimizer.solve()        │
                    │  (any of 200+ algorithms)   │
                    └──────────────┬──────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
     ┌────────▼────────┐ ┌────────▼────────┐ ┌────────▼────────┐
     │  Swarm-Based    │ │  Evolution-Based│ │  Physics-Based  │
     │  PSO, GWO, ABC  │ │  GA, DE, CMA-ES │ │  SA, GSA, MVO   │
     │  WOA, MFO, SCA  │ │  ES, EP, SHADE  │ │  EO, ASO, HGSO  │
     └─────────────────┘ └─────────────────┘ └─────────────────┘
              │                    │                    │
              └────────────────────┼────────────────────┘
                                   │
                        ┌──────────▼───────────┐
                        │    Best Solution      │
                        │  position + fitness   │
                        └──────────────────────┘
```

### Algorithm Families in MealPy

MealPy organizes its 200+ algorithms into families based on their inspiration source:

| Family | Inspiration | Examples | Count |
|--------|------------|----------|-------|
| **Swarm** | Animal group behavior | PSO (birds), GWO (wolves), WOA (whales), ABC (bees), FA (fireflies) | ~50 |
| **Evolution** | Biological evolution | GA, DE, CMA-ES, SHADE, EP | ~20 |
| **Physics** | Physical laws | Simulated Annealing, Gravitational Search, Electromagnetic | ~25 |
| **Human** | Human behavior | Teaching-Learning (TLBO), Brain Storm, Political Optimizer | ~15 |
| **Bio** | Biological processes | Virus Colony Search, Bacterial Foraging, Biogeography | ~15 |
| **System** | Systems/processes | Water Cycle, Ecosystem, Supply-Demand | ~10 |
| **Math** | Mathematical concepts | Sine-Cosine, Arithmetic Optimization, Golden Ratio | ~15 |
| **Music** | Musical processes | Harmony Search | ~5 |
| **Misc** | Various | Chaos Game, Social Ski Driver | ~20+ |

### How to Use MealPy (Code Walkthrough)

#### Step 1: Define Your Problem

```python
import numpy as np
from mealpy import FloatVar, IntegerVar, StringVar, BoolVar

# The objective function — what you want to minimize
# For HPO: this trains a model and returns RMSE
def objective_function(solution):
    """
    solution: a list of continuous values in [0, 1] range
    MealPy maps these to your actual parameter ranges internally
    """
    n_blocks    = int(solution[0])      # decoded from continuous
    n_filters   = int(solution[1])      # decoded from continuous
    kernel_size = int(solution[2])      # decoded from continuous
    dropout     = solution[3]           # already continuous
    lr          = solution[4]           # already continuous
    
    # Build and train your model
    model = build_tcn(n_blocks, n_filters, kernel_size, dropout, lr)
    history = model.fit(X_train, y_train, ...)
    rmse = evaluate(model, X_val, y_val)
    
    return rmse   # MealPy will minimize this

# Define bounds for each parameter
bounds = [
    IntegerVar(lb=2, ub=6, name="n_blocks"),
    IntegerVar(lb=32, ub=256, name="n_filters"),   # Note: can't specify {32,64,128,256}
    IntegerVar(lb=2, ub=7, name="kernel_size"),
    FloatVar(lb=0.05, ub=0.40, name="dropout"),
    FloatVar(lb=0.0001, ub=0.01, name="lr"),
]

# Create problem dictionary
problem = {
    "bounds": bounds,
    "obj_func": objective_function,
    "minmax": "min",          # minimize RMSE
    "log_to": "console",     # print progress
}
```

#### Step 2: Choose and Configure an Optimizer

```python
from mealpy import GWO, PSO, GA, DE, WOA, SCA

# Grey Wolf Optimizer
optimizer = GWO.OriginalGWO(epoch=30, pop_size=20)

# Or swap to any other algorithm — same interface:
# optimizer = PSO.OriginalPSO(epoch=30, pop_size=20, c1=2.0, c2=2.0, w=0.9)
# optimizer = GA.BaseGA(epoch=30, pop_size=20, pc=0.9, pm=0.1)
# optimizer = WOA.OriginalWOA(epoch=30, pop_size=20)    # Whale Optimization
# optimizer = DE.OriginalDE(epoch=30, pop_size=20)       # Differential Evolution

# That's the power of MealPy — swap one line, get a different algorithm
```

#### Step 3: Run and Get Results

```python
# Run the optimization
best_agent = optimizer.solve(problem)

# Extract results
best_position = best_agent.solution        # [3, 128, 5, 0.12, 0.0023]
best_fitness  = best_agent.target.fitness   # 0.0268

print(f"Best n_blocks:    {int(best_position[0])}")
print(f"Best n_filters:   {int(best_position[1])}")
print(f"Best kernel_size: {int(best_position[2])}")
print(f"Best dropout:     {best_position[3]:.4f}")
print(f"Best lr:          {best_position[4]:.6f}")
print(f"Best RMSE:        {best_fitness:.6f}")

# Access optimization history
print(f"Convergence: {optimizer.history.list_global_best_fit}")
```

#### Step 4: Compare Multiple Algorithms

```python
from mealpy import GWO, PSO, GA, WOA, SCA

algorithms = {
    "GWO":  GWO.OriginalGWO(epoch=30, pop_size=20),
    "PSO":  PSO.OriginalPSO(epoch=30, pop_size=20),
    "GA":   GA.BaseGA(epoch=30, pop_size=20, pc=0.9, pm=0.05),
    "WOA":  WOA.OriginalWOA(epoch=30, pop_size=20),
    "SCA":  SCA.OriginalSCA(epoch=30, pop_size=20),
}

results = {}
for name, opt in algorithms.items():
    best = opt.solve(problem)
    results[name] = best.target.fitness
    print(f"{name}: Best RMSE = {best.target.fitness:.6f}")
```

### MealPy's Internal Flow

When you call `optimizer.solve(problem)`, here's what happens inside:

```
solve(problem)
│
├─ 1. decode bounds → internal continuous representation
│     IntegerVar(2, 6)  → FloatVar(2.0, 6.0) + rounding
│     StringVar(["a","b","c"]) → FloatVar(0, 2) + index mapping
│
├─ 2. generate_population(pop_size)
│     Create N random positions within bounds
│     Evaluate each → N calls to obj_func
│
├─ 3. for epoch in range(max_epoch):
│     │
│     ├─ evolve(population)          ← algorithm-specific logic
│     │   GWO: wolves move toward alpha/beta/delta
│     │   PSO: particles update velocity + position
│     │   GA:  select parents → crossover → mutate
│     │
│     ├─ amend_position(population)  ← clip to bounds
│     │   Ensure all positions within [lb, ub]
│     │   Round integer variables
│     │
│     ├─ evaluate(population)        ← N calls to obj_func
│     │   Each call = full model training!
│     │
│     └─ update_global_best()
│         Track best solution across all epochs
│
└─ 4. return global_best
```

### The Categorical Problem in MealPy

MealPy's biggest limitation for deep learning HPO — handling categorical parameters:

```python
# What we WANT:
n_filters ∈ {32, 64, 128, 256}     # 4 specific choices, not a range

# What MealPy gives us:
IntegerVar(lb=32, ub=256)           # Any integer 32-256
# GWO might propose n_filters=87 or n_filters=193 — meaningless values

# Workaround 1: Map indices
IntegerVar(lb=0, ub=3)             # 0→32, 1→64, 2→128, 3→256
# But GWO treats 0→1 as same distance as 2→3
# In reality: 32→64 is doubling, 128→256 is also doubling

# Workaround 2: Use StringVar (MealPy ≥ 3.0)
StringVar(valid_sets=("32", "64", "128", "256"), name="n_filters")
# Better, but internally still maps to continuous [0, 3] and rounds
```

**Optuna handles this natively:**
```python
# Optuna — categorical is first-class
n_filters = trial.suggest_categorical("n_filters", [32, 64, 128, 256])
# TPE builds separate probability distributions per category
# No distance metric needed — each option is independent
```

### When MealPy Shines

MealPy is **best used for:**

| Use Case | Why MealPy Works |
|----------|-----------------|
| **Benchmarking algorithms** | 200+ algorithms, same API. Run all on your problem, compare. |
| **Continuous optimization** | All algorithms natively operate on continuous spaces |
| **Cheap objective functions** | Eval in seconds → 600 evals = minutes, not days |
| **Research** | Need to compare GWO vs. improved-GWO vs. hybrid-GWO? All available. |
| **Engineering problems** | PID tuning, structural optimization, antenna design — continuous + cheap |
| **Feature selection** | Binary encoding with transfer functions built in |
| **Combinatorial problems** | TSP, scheduling — with appropriate problem formulation |

MealPy is **NOT ideal for:**

| Use Case | Why Not |
|----------|---------|
| **Deep learning HPO** | Each eval = minutes. 600 evals = 100 hours. |
| **Mixed categorical + continuous** | Continuous engines struggle with categorical |
| **Need pruning/early stopping** | No mechanism to kill bad evaluations mid-training |
| **Need crash recovery** | No built-in checkpoint/resume like Optuna's SQLite |
| **Limited budget (<50 evals)** | Population methods need 100s of evals to converge |

### MealPy vs. Optuna: API Comparison

```python
# ═══════════════ OPTUNA ═══════════════
import optuna

def objective(trial):
    n_blocks  = trial.suggest_int("n_blocks", 2, 6)
    n_filters = trial.suggest_categorical("n_filters", [32, 64, 128, 256])
    kernel    = trial.suggest_categorical("kernel", [2, 3, 5, 7])
    dropout   = trial.suggest_float("dropout", 0.05, 0.40)
    lr        = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    
    model = build_tcn(n_blocks, n_filters, kernel, dropout, lr)
    model.fit(X_train, y_train, epochs=200,
              callbacks=[optuna.integration.TFKerasPruningCallback(trial, "val_loss")])
    
    return evaluate(model, X_val, y_val)

study = optuna.create_study(direction="minimize",
                            storage="sqlite:///hpo.db")  # crash recovery!
study.optimize(objective, n_trials=50, timeout=14400)

# Features used: categorical params, log scale, pruning, SQLite resume

# ═══════════════ MEALPY (GWO) ═══════════════
from mealpy import GWO, FloatVar, IntegerVar

def objective(solution):
    n_blocks  = int(solution[0])
    n_filters = [32, 64, 128, 256][int(solution[1])]  # manual mapping
    kernel    = [2, 3, 5, 7][int(solution[2])]         # manual mapping
    dropout   = solution[3]
    lr        = 10 ** solution[4]                       # manual log scale
    
    model = build_tcn(n_blocks, n_filters, kernel, dropout, lr)
    model.fit(X_train, y_train, epochs=200)  # NO pruning — trains fully
    
    return evaluate(model, X_val, y_val)

problem = {
    "bounds": [
        IntegerVar(lb=2, ub=6),
        IntegerVar(lb=0, ub=3),           # index into filter list
        IntegerVar(lb=0, ub=3),           # index into kernel list
        FloatVar(lb=0.05, ub=0.40),
        FloatVar(lb=-4, ub=-2),           # log10(lr)
    ],
    "obj_func": objective,
    "minmax": "min",
}

optimizer = GWO.OriginalGWO(epoch=30, pop_size=20)  # 600 evals!
best = optimizer.solve(problem)
# No crash recovery, no pruning, 12× more evaluations
```

### GWO Variants in MealPy

MealPy provides several improved versions of GWO:

| Variant | Key Improvement | When to Use |
|---------|----------------|-------------|
| `GWO.OriginalGWO` | Standard algorithm (Mirjalili 2014) | Baseline |
| `GWO.RW_GWO` | Random walk for better exploration | Multimodal landscapes |
| `GWO.IGWO` | Improved GWO with opposition-based learning | Faster convergence needed |

```python
from mealpy import GWO

# Original
opt1 = GWO.OriginalGWO(epoch=50, pop_size=30)

# Random Walk variant
opt2 = GWO.RW_GWO(epoch=50, pop_size=30)
```

### Bottom Line: GWO + MealPy

**GWO** is an elegant algorithm — simple math, few hyperparameters, smooth exploration-to-exploitation transition. It excels on continuous, cheap-to-evaluate, multimodal problems (engineering design, control systems, signal processing).

**MealPy** is a powerful library — 200+ algorithms, unified API, great for research and benchmarking. It makes metaheuristic experimentation trivial.

**But for deep learning HPO** (expensive evaluations, mixed parameter types, need for pruning and crash recovery), **Optuna's TPE remains the better tool**. The evaluation budget gap (50 vs. 600 trials) is the decisive factor when each trial costs minutes of GPU time.

---

## Head-to-Head Comparison

### Evaluation Cost

| Method | Population | Iterations | Total Evals | At 10 min/eval |
|--------|-----------|------------|-------------|----------------|
| **TPE (Optuna)** | 1 | 50 | **50** | **~8 hours** |
| **GWO (mealpy)** | 20 | 30 | **600** | **~100 hours** |
| **PSO (mealpy)** | 20 | 30 | **600** | **~100 hours** |
| **GA (mealpy)** | 30 | 20 | **600** | **~100 hours** |
| **Random Search** | 1 | 100 | **100** | **~17 hours** |
| **Grid Search** | 1 | all | **5,120+** | **~850 hours** |

### Feature Comparison

| Feature | TPE (Optuna) | GWO (mealpy) | PSO (mealpy) | GA (mealpy) | Hyperband |
|---------|-------------|-------------|-------------|-------------|-----------|
| **Sample efficiency** | ★★★★★ | ★★ | ★★ | ★★★ | ★★★★ |
| **Categorical params** | ★★★★★ | ★★ | ★★ | ★★★★ | ★★★★★ |
| **Mixed types** | ★★★★★ | ★★ | ★★ | ★★★ | ★★★★★ |
| **Global exploration** | ★★★ | ★★★★★ | ★★★★ | ★★★★★ | ★★★ |
| **Escaping local optima** | ★★★ | ★★★★★ | ★★★★ | ★★★★★ | ★★ |
| **Crash recovery** | ★★★★★ | ★ | ★ | ★ | ★★★★★ |
| **Pruning (early kill)** | ★★★★★ | ✗ | ✗ | ✗ | ★★★★★ |
| **Parallelization** | ★★★ | ★★★★★ | ★★★★★ | ★★★★★ | ★★★★★ |
| **No objective formula needed** | ✓ | ✓ | ✓ | ✓ | ✓ |

### Quality vs Budget

```
RMSE ↓
│
│  Grid ●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━●  (best possible, infinite budget)
│                              TPE ●━━━━━━━━━━━●  (near-optimal in 50 trials)
│                     GA ●━━━━━━━━━●               (good in 600 trials)
│                   GWO ●━━━━━━━━●                 (good in 600 trials)
│          Random ●━━━━━●                          (decent in 100 trials)
│
└──────────────────────────────────────────────── Budget (evaluations) →
     10        50       100      200      500    1000
```

---

## Why TPE (Optuna) for Our TCN

Our specific constraints:

| Constraint | Impact |
|-----------|--------|
| **Each eval = 5–30 min** | Can't afford 600 evals. Need <50. |
| **6 hyperparameters** | Moderate dimensionality — TPE handles fine |
| **4 categorical params** | GWO/PSO need awkward encoding |
| **Kaggle kernel = 12 hours max** | Budget-limited, need every eval to count |
| **Kernel crashes happen** | Need resumable storage |
| **Bad configs waste GPU time** | Need pruning to kill them early |

**TPE (Optuna) wins on every constraint.**

### What Would Need to Change for GWO to Be Better

| Change | Why It Helps GWO |
|--------|-----------------|
| Eval time drops to <10 seconds | Population of 20 × 50 iterations = 1000 evals in 2.8 hours |
| All params become continuous | No categorical encoding problem |
| Landscape has many local optima | GWO's population explores more globally |
| Unlimited compute budget | Population methods converge to true global optimum |
| Need parallelization across 20+ GPUs | Each wolf = 1 GPU, natural fit |

---

## When Metaheuristics Win

### Scenario 1: Feature Selection
- **Task:** Pick best 50 features from 500
- **Search space:** Binary vector of length 500 (include/exclude)
- **Eval time:** Seconds (train a simple model)
- **Winner:** GA, PSO — binary encoding is natural, evals are cheap

### Scenario 2: Neural Architecture Search (NAS) with Cluster
- **Task:** Design network topology (layers, connections, operations)
- **Eval time:** 30 minutes per architecture
- **Compute:** 100 GPUs available
- **Winner:** GA — 100 wolves evaluate in parallel, population explores diverse architectures

### Scenario 3: Control System Tuning
- **Task:** Tune PID controller gains (3 continuous params)
- **Eval time:** 0.1 seconds (simulation)
- **Landscape:** Many local optima
- **Winner:** GWO, PSO — smooth continuous space, cheap evals, global exploration needed

### Scenario 4: HPO for Deep Learning on Single GPU
- **Task:** Tune TCN/LSTM/Transformer hyperparameters
- **Eval time:** 10+ minutes
- **Compute:** 1-2 GPUs
- **Winner:** TPE (Optuna) — sample efficient, pruning saves GPU time, crash recovery

---

## Decision Flowchart

```
Is each evaluation expensive (>1 min)?
├── YES → Do you need >100 evaluations to cover the space?
│         ├── YES → Do you have many GPUs for parallel eval?
│         │         ├── YES → Population-based (GA/GWO) with parallel eval
│         │         └── NO  → TPE (Optuna) with pruning ← OUR CASE
│         └── NO  → TPE (Optuna) — most sample-efficient
└── NO  → Are parameters mostly continuous?
          ├── YES → PSO or GWO (fast, good global search)
          └── NO  → GA (handles mixed/discrete types naturally)

Do you have categorical parameters?
├── YES → TPE or GA (native support)
└── NO  → Any method works

Do you need crash recovery?
├── YES → Optuna (SQLite storage)
└── NO  → Any method works

Is the landscape full of local optima?
├── YES → Population methods (GWO, GA, PSO) or
│         Optuna with high exploration (large n_startup_trials)
└── NO  → TPE converges fastest
```

---

## Summary Table

| Your Question | Answer |
|--------------|--------|
| Why not GWO? | 600+ evals × 10 min = 100 hours. We have 4 hours. |
| Is GWO bad? | No — wrong tool for this job. Great for cheap, continuous problems. |
| Could we use GA? | Best metaheuristic option, but still needs 300+ evals minimum. |
| What about PSO? | Same issue — population × iterations = too many evals. |
| Best for our case? | TPE (Optuna) — 50 trials, pruning, mixed params, crash recovery. |
| When to switch to GWO? | If we get eval time under 10 seconds or access to 20+ parallel GPUs. |

---

*Document prepared August 12, 2026 — Updated August 13, 2026 — Ibrahim Hanafy, NU Research Internship*
