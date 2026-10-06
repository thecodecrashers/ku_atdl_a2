# Computing Nonvacuous Generalization Bounds for Deep (Stochastic) Neural Networks with Many More Parameters than Training Data — PyTorch port

This is a port of the PAC-Bayes generalization bound optimization for stochastic neural networks, as described in the article "[Computing Nonvacuous Generalization Bounds for Deep (Stochastic) Neural Networks with Many More Parameters than Training Data](https://arxiv.org/pdf/1703.11008.pdf)" by Dziugaite and Roy, published in *Uncertainty in AI* (2017).

The original implementation (Python 3.5, TensorFlow 1.10, Keras 2.2) has been migrated to **modern PyTorch (>= 2.0) and Python 3.11** (it also works on newer 3.x versions). The algorithm, the project layout, the command line interface, the hyper-parameters and the checkpoint/output file formats are unchanged.

## Requirements
Python 3.11 (3.9+ should work), `torch>=2.0`, `numpy>=1.24`:

```
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Run all commands below from the root of this folder (the one containing `snn/` and `mnist/`). A GPU is used automatically if available (`--device cpu` / `--device cuda:0` to force one).

## Instructions

We provide a streamlined experiment runner to easily reproduce the results from the paper and run specific ablations.

### 1. Run Baseline Reproduction (Table 1)
To run all baseline architectures (T-600, T-1200, T-300x2, T-600x2, T-1200x2, T-600x3, R-600) for Table 1:
```bash
python experiments/run_experiments.py --suite baseline
```

### 2. Run Robustness Ablations
To run additional PAC-Bayes ablations (reusing the T-600 checkpoint from the baseline suite):
```bash
python experiments/run_experiments.py --suite ablation
```

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
