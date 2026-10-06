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

3. Run the baseline suite (this will reproduce Table 1 from the paper):
   ```bash
   python experiments/run_experiments.py --suite baseline
   ```

4. Run the robustness ablations (this reuses the T-600 checkpoint to test various PAC-Bayes hyperparameters):
   ```bash
   python experiments/run_experiments.py --suite ablation
   ```

5. Plot the results:
   ```bash
   python experiments/plot_results.py
   ```
   This will output a `table1_comparison.csv` and history plots inside the `results/` folder.

For full details regarding the implementation, hyperparameters, and reproduction details, please read:
- [PAC_Bayes_modeifeid/SUMMARY.md](PAC_Bayes_modeifeid/SUMMARY.md)
- [PAC_Bayes_modeifeid/REPRODUCTION_NOTES.md](PAC_Bayes_modeifeid/REPRODUCTION_NOTES.md)
