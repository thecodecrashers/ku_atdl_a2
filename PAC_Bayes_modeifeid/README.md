# Computing Nonvacuous Generalization Bounds for Deep (Stochastic) Neural Networks with Many More Parameters than Training Data — PyTorch port

This is a port of the PAC-Bayes generalization bound optimization for stochastic neural networks, as described in the article "[Computing Nonvacuous Generalization Bounds for Deep (Stochastic) Neural Networks with Many More Parameters than Training Data](https://arxiv.org/pdf/1703.11008.pdf)" by Dziugaite and Roy, published in *Uncertainty in AI* (2017).

The original implementation (Python 3.5, TensorFlow 1.10, Keras 2.2) has been migrated to **modern PyTorch (>= 2.0) and Python 3.11** (it also works on newer 3.x versions). The port retains the original network parameter layout and legacy checkpoint format. The experiment runner provides baseline runs and the three ablations described below.

The corrected implementation computes KL using the current trainable mean, constrains the prior to the positive-index variance grid, and applies both Monte Carlo and PAC-Bayes KL inversions during final evaluation. See [BOUND_FIXES.md](BOUND_FIXES.md) for the formulas, confidence interpretation, and validation. Results from the earlier objective require a new second-stage run.

## Requirements
Python 3.11 (3.9+ should work), `torch>=2.0`, `numpy>=1.24`:

```
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Run all commands below from the root of this folder (the one containing `snn/` and `mnist/`). A GPU is used automatically if available (`--device cpu` / `--device cuda:0` to force one).

## Instructions

We provide an experiment runner for the paper's baseline architectures and the initialization/frozen-mean ablations. This does not claim exact reproduction of every Table 1 setting; see [REPRODUCTION_NOTES.md](REPRODUCTION_NOTES.md) for differences. MNIST experiments require the four IDX `.gz` files in `mnist/`; datasets, checkpoints, and local results are excluded from Git.

### 1. Run Baseline Reproduction (Table 1)
To run all baseline architectures (T-600, T-1200, T-300x2, T-600x2, T-1200x2, T-600x3, R-600) for Table 1:
```bash
python experiments/run_experiments.py --suite baseline
```

Each baseline invocation creates a unique run root. `--output_dir <new-directory>` or `PACB_RUN_ROOT` can select a new root explicitly; existing roots are rejected. For T-600 alone with 1,000 independent posterior draws and final-only evaluation:

```bash
python experiments/run_experiments.py --experiment T600 --snn_samples 1000 --eval_interval 0
```

To match the upstream code's default training seed, use `--seed 11` explicitly
for both T600 and the three ablations. The paper does not list the Table 1
training seeds; the same numeric seed also does not produce identical random
draws across TensorFlow and PyTorch. Within this PyTorch version, the commands
below use the same initial weights and fixed prior mean across the four runs:

```bash
python -u experiments/run_experiments.py --experiment T600 --seed 11 --snn_samples 150000 --eval_interval 0
python -u experiments/run_ablation_fast.py --seed 11 --sgd_steps 1 --frozen_sgd_epochs 20 --snn_samples 150000 --eval_interval 0 --progress_interval 15
```

These commands start new training runs in fresh result directories. The three
ablations run concurrently; the baseline is a separate invocation. Training
uses seed 11, while final Monte Carlo draws use an independent recorded seed
for the confidence correction. Without an explicit `--seed`, baseline runs
still use 11 and ablations still select a fresh seed.

### 2. Run the Three Ablations
```bash
python experiments/run_ablation_fast.py
```

- **Random initialization (`random_init`):** Skip SGD and initialize the second-stage PAC-Bayes optimization with freshly sampled random weights.
- **SGD initialization (`sgd_init`):** Perform one SGD mini-batch update by default, then initialize the second stage with the resulting weights.
- **Frozen SGD weights (`no_trainw`):** Train SGD for 20 epochs, then keep every weight and bias fixed during the second stage. Optimize only the posterior variances and prior variance.

The first two conditions optimize the posterior mean, posterior variances, and prior variance during the second stage; the third freezes the posterior mean. All three share one initialization seed within each run. By default it is fresh and independent of T-600 baseline seeds; an explicit `--seed 11` aligns the numeric training seed with the baseline. No existing baseline checkpoint is loaded. Use `--seed <integer>` for an explicitly reproducible run.

Three threads launch separate training processes. Each invocation creates a new output directory with separate `run.log` files, preserving existing results. Training accuracy evaluation is disabled by default (`--eval_interval 0`); final train/test evaluation is always performed. Use `--eval_interval 50` for diagnostics every 50 epochs.

The terminal reports each condition's SGD, PAC-Bayes, and final evaluation stage, completed updates, elapsed time, and estimated time remaining for the current stage. Progress is refreshed every 30 seconds by default; use `--progress_interval 10` for more frequent updates. ETA uses recent observed throughput and becomes available after two progress measurements. Full child output is still retained in each `run.log`; failures also show the last log lines in the terminal.

`--snn_samples 1` remains the fast default. It gives a valid but usually loose bound after the Monte Carlo correction. Use `--snn_samples 1000` for a more useful final estimate; this increases final evaluation time. `SNN train error` and `SNN test error` are sampled errors, while `SNN train error upper bound` is the Monte Carlo confidence bound used in `PAC-Bayes bound`. The final bound has failure probability `0.025 + 0.01 = 0.035` for each experiment.

New model files contain a complete `final_posterior` record. Reevaluation writes a separate file and refuses legacy outputs with incomplete final snapshots:

```bash
python experiments/recompute_summary.py --output_dir <completed-run-directory> --snn_samples 1000
```

For the paper's 150,000 independent posterior draws, choose a fresh output file:

```bash
python experiments/recompute_summary.py --output_dir <completed-run-directory> --snn_samples 150000 --summary_path <new-summary-file.json>
```

The confidence stays at 96.5% per experiment because the failure probabilities stay fixed; a larger sample count reduces the Monte Carlo correction. `run_mc_comparison.py` automates comparisons for the locally configured saved runs. Its archived input paths must exist; it does not train missing models in a fresh clone.

To use one full SGD epoch instead of one mini-batch update:
```bash
python experiments/run_ablation_fast.py --sgd_epochs 1
```
For `sgd_init`, use `--sgd_steps <N>` for a specified number of mini-batch updates, or `--sgd_epochs 20` for 20 complete epochs. These options are mutually exclusive. They do not change the frozen-weight pretraining, which uses `--frozen_sgd_epochs 20` by default. The standalone `run_ablation_only.py` accepts the same options, and `run_experiments.py --suite ablation` delegates to this suite.

Zero biases stay unchanged; only the initial posterior standard deviation has a floor of `1e-8` to avoid `log(0)`. All three processes may share the selected GPU, so concurrent execution does not guarantee higher GPU throughput.

### 3. Generate Tables and Plots
After running the suites, you can aggregate the results and generate evaluation plots by running:
```bash
python experiments/plot_results.py
```
This will output `table1_comparison.csv` and history plots inside the `results/` directory.

> **Note:** For more information about the reproduction details and specific ablation protocols, please read [SUMMARY.md](SUMMARY.md) and [REPRODUCTION_NOTES.md](REPRODUCTION_NOTES.md).

### (Optional) Manual Execution
You can still run individual scripts manually exactly as in the original implementation:
```bash
# SGD Phase
python snn/experiments/run_sgd.py fc --layers 600 --sgd_epochs 20 --binary

# PAC-Bayes Phase
python snn/experiments/run_pacb.py fc --layers 600 --sgd_epochs 20 --pacb_epochs 1000 --lr 0.001 --drop_lr 250 --lr_factor 0.1 --binary
```

## What changed compared to the TensorFlow version
| Original (TF 1.x / Keras) | This port (PyTorch) |
|---|---|
| `tf.Graph` / `tf.Session` / placeholders / `variable_scope` | Eager PyTorch tensors; parameters are a list of leaf tensors in the same order and layout as before (`W` is `[n_in, n_out]`, conv kernels are `[H, W, in, out]`) |
| `tf.train.MomentumOptimizer` | `torch.optim.SGD(momentum=0.9)` (identical update rule) |
| `tf.train.RMSPropOptimizer` | `snn/core/optim.py::TFRMSprop`, reproducing TF's RMSProp exactly (decay 0.9, accumulator initialised to 1, `eps` inside the sqrt) |
| `tensorflow.examples.tutorials.mnist` | Plain numpy reader of the `.gz` files in `mnist/` (first 5000 training images are still held out, so the training set has 55000 images as before) |
| `keras.datasets.cifar10` | Plain numpy/urllib loader in `snn/core/data_fn.py` |
| `tf.nn.lrn`, `SAME` padded max-pool | `F.local_response_norm` / padded `F.max_pool2d`, configured to be numerically equivalent |

Checkpoints written by the original code (e.g. `snn/experiments/binary_mnist/FC_layers[600]_epochs20_seed11.pickle`, included) are lists of numpy arrays and can be loaded directly by `run_pacb.py`. Random numbers come from PyTorch/NumPy instead of TensorFlow, so a fresh run will not be bit-identical to one from the old code.

Small fixes needed to run on current libraries: the `CNN` model no longer crashes because of `layers=None`, the multi-class case is handled when counting correct predictions during PAC-Bayes optimization (the binary case is unchanged), and the `sys.path` hacks in the experiment scripts were replaced by a path relative to the repository.

## Acknowledgments

The development of this code was initiated by [Gintare Karolina Dziugaite](https://gkdz.org) and [Daniel M. Roy](http://danroy.org), while they were visiting the Simons Institute for the Theory of Computing at U.C. Berkeley. During this course of this research project, GKD was supported by an EPSRC studentship; DMR was supported by an NSERC Discovery Grant, Connaught Award, and U.S. Air Force Office of Scientific Research grant #FA9550-15-1-0074.

Waseem Gharbieh (Element AI) and Gabriel Arpino (University of Toronto) contributed to improving and testing the code, and helped produce this code release.

## BiBTeX

    @inproceedings{DR17,
            title = {Computing Nonvacuous Generalization Bounds for Deep (Stochastic) Neural Networks with Many More Parameters than Training Data},
           author = {Gintare Karolina Dziugaite and Daniel M. Roy},
             year = {2017},
        booktitle = {Proceedings of the 33rd Annual Conference on Uncertainty in Artificial Intelligence (UAI)},
    archivePrefix = {arXiv},
           eprint = {1703.11008},
    }
