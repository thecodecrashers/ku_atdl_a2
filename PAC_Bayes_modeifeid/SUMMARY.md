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
   - Retained `--snn_samples=1` as a fast option and added the Monte Carlo confidence correction. Use more samples for tighter final bounds; one draw does not match Table 1 precision.
   - Corrected the current-mean KL gradient, prior variance domain, confidence parameters, and complete final posterior serialization. See [BOUND_FIXES.md](BOUND_FIXES.md).

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

### 3. Run Three Ablations

```bash
python experiments/run_ablation_fast.py
```

The suite runs three conditions concurrently:
- **Random initialization:** Initialize the PAC-Bayes stage from fresh random weights without any SGD updates.
- **SGD initialization:** Initialize the PAC-Bayes stage from weights obtained after one SGD mini-batch update by default.
- **Frozen SGD weights:** Pretrain with SGD for 20 epochs, then optimize variances while keeping all posterior mean weights and biases fixed.

Only the first two conditions train the posterior mean in the second stage. All three share a fresh initialization seed within the run and do not reuse baseline checkpoints or seeds. For `sgd_init`, use `--sgd_epochs 1` for a complete SGD epoch or `--sgd_epochs 20` for full pretraining instead of a single update. The frozen-weight condition has an independent `--frozen_sgd_epochs` option, defaulting to 20.

Each invocation writes to a new directory. The default remains 1,000 PAC-Bayes epochs, final evaluation only, and `snn_samples=1`. `run_ablation_only.py` and the `--suite ablation` entry point call the same suite.

### 4. Plot and Extract Results

After running the suites, you can aggregate the results and generate plots:

```bash
python experiments/plot_results.py
```

This produces:
- `results/table1_comparison.csv`
- `results/t600_pac_bound_vs_epoch.png`
- `results/t600_kl_vs_epoch.png`
