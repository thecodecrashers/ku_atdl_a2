# Project Summary: PAC-Bayes Bounds for Deep Stochastic Networks

This repository contains a modernized, reproducible PyTorch implementation of the PAC-Bayes bound optimization method from **"Computing Nonvacuous Generalization Bounds for Deep (Stochastic) Neural Networks with Many More Parameters than Training Data" (Dziugaite & Roy, 2017)**.

## Project Structure

- `snn/`: The core PAC-Bayes optimization library.
  - `core/network.py`: The neural network interface, SGD loop, and PAC-Bayes optimization logic.
  - `core/optim.py`: A PyTorch reimplementation of `tf.train.RMSPropOptimizer` to match exact TensorFlow 1.x behavior.
  - `experiments/run_sgd.py`: SGD pre-training script.
  - `experiments/run_pacb.py`: PAC-Bayes bound optimization script.
- `experiments/`: Orchestration and plotting scripts.
  - `run_experiments.py`: The main experiment orchestrator. Handles intelligent checkpoint reuse, isolated run directories, and suite execution (`baseline`, `ablation`).
  - `plot_results.py`: A utility script to aggregate JSON summaries, output `table1_comparison.csv`, and plot the PAC bound history and KL divergence.
- `results/`: Output directory structure where all logs, histories, and serialized models are saved.

## Key Enhancements

We have cleanly separated the mathematical core of the repository from the experiment orchestration logic.

1. **Robust Pipeline:** Added `run_experiments.py` to seamlessly execute and track experiments without manual checkpoint management.
2. **Comprehensive Logging:** Extracted the PAC-Bayes loss, objective, KL divergence, and stochastic accuracy at every epoch into `history.csv` files.
3. **Reproducibility Guarantees:** 
   - PyTorch `RMSProp` is mapped to legacy TF behavior.
   - Initializer is fixed to truncated normal to avoid divergence.
   - Added secure `--random_labels` for the R-600 ablation.
   - Introduced the exact `--snn_samples=1` metric from the paper's original open-source release to match `Table 1`.

## How to Run

### 1. Environment

First, ensure you are using the correct Python environment with PyTorch installed.

### 2. Run Baselines (Table 1 Reproduction)

Run all experiments corresponding to Table 1 from the paper:

```bash
python experiments/run_experiments.py --suite baseline
```

This will run:
- `T600`
- `T1200`
- `T300x2`
- `T600x2`
- `T1200x2`
- `T600x3`
- `R600` (with `--random_labels`)

### 3. Run Ablations

Run the robust evaluation ablations (which will automatically reuse the existing `T600` checkpoint from the baseline suite!):

```bash
python experiments/run_experiments.py --suite ablation
```

This tests:
- **Ablation A:** 5,000 PAC-Bayes epochs instead of 1,000.
- **Ablation B:** PAC-Bayes with frozen posterior mean (`--no-trainw`).
- **Ablation C:** True Monte Carlo Stochastic Neural Network evaluation with `N=200` instead of `N=1`.

### 4. Plot and Extract Results

After running the suites, you can aggregate the results and generate plots:

```bash
python experiments/plot_results.py
```

This produces:
- `results/table1_comparison.csv`
- `results/t600_pac_bound_vs_epoch.png`
- `results/t600_kl_vs_epoch.png`
