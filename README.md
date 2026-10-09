# PAC-Bayes Bounds for Deep (Stochastic) Neural Networks

This repository contains two folders:
1. `pacbayes-opt/`: The original TensorFlow implementation from the paper "Computing Nonvacuous Generalization Bounds for Deep (Stochastic) Neural Networks with Many More Parameters than Training Data" (Dziugaite & Roy, 2017).
2. `PAC_Bayes_modeifeid/`: A modernized, reproducible PyTorch port of the method.

## Running the PyTorch Implementation

All of the automated reproducibility logic is located in `PAC_Bayes_modeifeid`.

1. Go into the folder:
   ```bash
   cd PAC_Bayes_modeifeid
   ```

2. (Optional) Set up a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate          # Windows
   # source .venv/bin/activate     # Linux/Mac
   pip install -r requirements.txt
   ```

3. Run the baseline architectures corresponding to Table 1:
   ```bash
   python experiments/run_experiments.py --suite baseline --snn_samples 1000 --eval_interval 0
   ```

4. Run all three ablations concurrently: random initialization, SGD initialization, and frozen SGD weights:
   ```bash
   python experiments/run_ablation_fast.py --snn_samples 1000
   ```

   The first two conditions train the posterior mean during the PAC-Bayes stage. The third keeps the 20-epoch SGD weights fixed. By default all three share a fresh seed independent of the baseline and write to a new directory. Add `--seed 11` to match the baseline's numeric training seed. Use `--sgd_epochs 1` for a complete SGD epoch in the SGD-initialized condition instead of one mini-batch update. `python experiments/run_ablation_only.py` runs the same three experiments.

5. Plot the results:
   ```bash
   python experiments/plot_results.py
   ```
   This will output a `table1_comparison.csv` and history plots inside the `results/` folder.

The final bound includes Monte Carlo uncertainty. The paper used 150,000 posterior samples; 1,000 samples usually give a larger correction at the same 96.5% per-experiment confidence. The default fast setting is one sample and typically gives a loose bound. Training schedules and the random-label protocol also differ from the paper, so these commands are not a claim of exact Table 1 reproduction.

To reevaluate a completed corrected model with the paper's sampling count, without retraining or overwriting its original summary:

```bash
python experiments/recompute_summary.py --output_dir <completed-run-directory> --snn_samples 150000 --summary_path <new-summary-file.json>
```

Datasets, model checkpoints, and local result directories are excluded from Git. Supply the MNIST IDX `.gz` files in `PAC_Bayes_modeifeid/mnist/` before running MNIST experiments.

For full details regarding the implementation, hyperparameters, and reproduction details, please read:
- [PAC_Bayes_modeifeid/SUMMARY.md](PAC_Bayes_modeifeid/SUMMARY.md)
- [PAC_Bayes_modeifeid/REPRODUCTION_NOTES.md](PAC_Bayes_modeifeid/REPRODUCTION_NOTES.md)
- [PAC_Bayes_modeifeid/BOUND_FIXES.md](PAC_Bayes_modeifeid/BOUND_FIXES.md)
