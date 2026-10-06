# Reproduction Notes

This document describes the changes and exact reproduction details for the PyTorch port of the PAC-Bayes margin optimization method by Dziugaite & Roy (2017).

## 1. Exact Baseline Match

To ensure our PyTorch port serves as a strict equivalent of the original TensorFlow implementation for the baseline, we verified the following settings:
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
- **Evaluation Samples:** As in the original `run_pacb.py` script, `N_SNN_samples=1` is used as the default for final stochastic network evaluation in the baseline suite.

## 2. Differences due to PyTorch Port
- **Optimization:** We created `snn/core/optim.py` with `TFRMSprop` to perfectly match the legacy `tf.train.RMSPropOptimizer`, since PyTorch's native `RMSprop` behaves slightly differently with epsilon placement and momentum.
- **Initialization:** We implemented `_trunc_normal` to mimic `tf.truncated_normal_initializer` which resampling values outside 2 standard deviations.
- **Data Iteration:** Handled via custom numpy batching, identical to the original logic, rather than `torch.utils.data.DataLoader`, to maintain perfect reproducibility with the original shuffled batches.

## 3. Experiment Pipeline Additions

The new experiment pipeline under `experiments/` allows automated and organized execution:
- **Command Line Arguments:** Added `--snn_samples`, `--output_dir`, `--run_name`, and robust boolean parsing for `--trainw` using `argparse.BooleanOptionalAction`.
- **Automated Runner:** `experiments/run_experiments.py` can run specific suites (`baseline`, `ablation`, `all`) or individual experiments (e.g. `T600`).
- **Checkpoint Reuse:** The runner securely reuses SGD checkpoints to save computation time across PAC-Bayes ablations sharing the same architecture and seed. Checkpoint reuse is strictly governed by matching hyperparameters.
- **Result Output:** All outputs (logs, metrics, bound history) are organized systematically into `results/baseline/` and `results/ablation/` rather than overwriting generic files.

## 4. Random Label Protocol (R-600)

For the `R-600` baseline experiment, the training labels are randomly shuffled (using a seeded permutation).
- **Implementation:** Added a `--random_labels` flag. When True, only the training set labels are randomly shuffled. The validation/test set labels are kept intact (though they are not used for optimization). This tests the PAC-Bayes bound behavior when generalizing is purely impossible (due to random labeling), verifying that the bound safely becomes non-vacuous or behaves as expected.
