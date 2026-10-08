# Reproduction Notes

This document describes the changes and exact reproduction details for the PyTorch port of the PAC-Bayes margin optimization method by Dziugaite & Roy (2017).

## 1. Baseline Settings

The port uses the following baseline settings. The corrected bound implementation follows the mathematical formulas and repairs inherited implementation issues described in [BOUND_FIXES.md](BOUND_FIXES.md).
- **Dataset:** Binary MNIST (digits 0-4 mapped to -1, digits 5-9 mapped to +1).
- **Training Set:** The first 5,000 examples are held out, leaving 55,000 examples for training.
- **Model Architecture:** Fully Connected (FC) networks with ReLU activations.
- **Weight Initialization:** Truncated-normal distribution with std `0.04`. Default bias initialization is `0.1` for the first layer and `0.0` for others.
- **SGD Optimizer (Pre-training):**
  - Learning Rate: `0.01`
  - Momentum: `0.9`
  - Batch Size: `100`
- **PAC-Bayes Optimizer:**
  - Objective: Evaluates the PAC-Bayes bound and minimizes it using a PyTorch equivalent of `tf.train.RMSPropOptimizer` (implemented as `TFRMSprop`).
- **Prior and Posterior Initialization:**
  - Prior `P`: Center initialized at `w0` (random weights before SGD), with optimized prior variance.
  - Posterior `Q`: Mean initialized at `w_sgd` (weights after SGD).
- **Evaluation Samples:** One independent posterior draw remains the fast default. The final bound now includes Monte Carlo uncertainty; one draw usually gives a loose bound. Use `--snn_samples 1000` for a more useful estimate. The paper used 150,000 samples, so the fast default does not reproduce the precision of Table 1.

## 2. Differences due to PyTorch Port
- **Optimization:** We created `snn/core/optim.py` with `TFRMSprop` to perfectly match the legacy `tf.train.RMSPropOptimizer`, since PyTorch's native `RMSprop` behaves slightly differently with epsilon placement and momentum.
- **Initialization:** We implemented `_trunc_normal` to mimic `tf.truncated_normal_initializer` which resampling values outside 2 standard deviations.
- **Zero Parameters:** The original zero biases and prior weights are preserved. The initial posterior standard deviation is `max(2*abs(w), 1e-8)` to keep its logarithm finite, including in the random-initialization ablation.
- **Data Iteration:** Custom NumPy batching shuffles before selecting each epoch's first batch. Framework-specific random streams prevent identical trajectories across TensorFlow and PyTorch.

## 3. Experiment Pipeline Additions

The new experiment pipeline under `experiments/` allows automated and organized execution:
- **Command Line Arguments:** Added `--snn_samples`, `--output_dir`, `--run_name`, and robust boolean parsing for `--trainw` using `argparse.BooleanOptionalAction`.
- **Automated Runner:** `experiments/run_experiments.py` can run specific suites (`baseline`, `ablation`, `all`) or individual experiments (e.g. `T600`).
- **Ablation Suite:** All ablation entry points run three conditions concurrently: random initialization, initialization after one SGD mini-batch update, and frozen weights after 20 SGD epochs. The first two train the posterior mean during PAC-Bayes optimization; the third does not. The `--sgd_epochs` option selects full epochs for `sgd_init`, while `--frozen_sgd_epochs` controls the third condition separately.
- **Independent Initialization:** No baseline checkpoint or seed is reused. Each run generates a fresh seed shared by all three conditions and writes new checkpoints and results to an isolated directory. An explicit `--seed` makes the suite reproducible.
- **Result Output:** Baselines and ablations use unique run roots. Existing summaries and models are protected, and reevaluation writes a separate summary.

## 4. Random Label Protocol (R-600)

For the `R-600` baseline experiment, the training labels are randomly shuffled (using a seeded permutation).
- **Implementation:** Added a `--random_labels` flag. When True, only the training set labels are randomly shuffled. Test labels are kept intact and are not used for optimization. This permutation protocol differs from the paper's independent random labels. The local runner also uses the common 20-epoch SGD and 1,000-epoch PAC schedule, rather than the paper's separate R-600 training budgets; it therefore does not establish an exact R-600 reproduction.
