# ATDL Assignment 2: PAC-Bayes Bounds for Stochastic Neural Networks

This repository contains two implementations of the PAC-Bayes generalization bound optimization for stochastic neural networks, as described in the seminal paper:
> **[Computing Nonvacuous Generalization Bounds for Deep (Stochastic) Neural Networks with Many More Parameters than Training Data](https://arxiv.org/abs/1703.11008)**  
> *Gintare Karolina Dziugaite and Daniel M. Roy (UAI 2017)*

## Repository Structure

- `pacbayes-opt/`: The original implementation provided by the authors (Python 3.5, TensorFlow 1.10, Keras 2.2).
- `PAC_Bayes_modeifeid/`: The modernized **PyTorch** port (Python 3.11+, PyTorch >= 2.0). It mathematically replicates the original TensorFlow computations exactly.

---

## How to Run the Modified PyTorch Version

The `PAC_Bayes_modeifeid` directory contains the updated, ready-to-run PyTorch version.

### 1. Environment Setup

It is highly recommended to use a virtual environment. The required packages are PyTorch and NumPy.

```bash
cd PAC_Bayes_modeifeid
python -m venv .venv
# Activate environment (Windows)
.venv\Scripts\activate
# Install dependencies
pip install -r requirements.txt
```

### 2. Running the Code (Based on Original Paper Parameters)

Following the experimental setup in Dziugaite and Roy (2017) for the **Binary MNIST** dataset, a fully-connected (FC) neural network with a single hidden layer of 600 neurons is optimized.

The process consists of two stages:

#### Stage 1: SGD Pre-training
First, we find a good initialization (a "prior" centered near a local minimum) by running standard SGD. 

```bash
# Train a 600-neuron FC network on Binary MNIST using SGD for 20 epochs
python snn/experiments/run_sgd.py fc --layers 600 --sgd_epochs 20 --binary
```
*Note: This will save the network weights to `snn/experiments/binary_mnist/FC_layers[600]_epochs20...pickle`.*

#### Stage 2: PAC-Bayes Bound Optimization
Next, we optimize the PAC-Bayes objective over the weights and variances, initialized from the SGD checkpoint. According to the original repository and paper, the optimization uses RMSProp for 1000 epochs, starting with a learning rate of 0.001 which decays by a factor of 0.1 after 250 epochs.

```bash
# Optimize the PAC-Bayes bound
python snn/experiments/run_pacb.py fc --layers 600 --sgd_epochs 20 --pacb_epochs 1000 --lr 0.001 --drop_lr 250 --lr_factor 0.1 --binary
```

### Expected Output
The `run_pacb.py` script will output the training statistics per epoch, and finally evaluate the deterministic and stochastic network accuracies, alongside the **Nonvacuous PAC-Bayes Bound**. You should see the PAC bound error strictly bounding the test error!

---
*Note: A CNN on CIFAR-10 is also supported (just replace `fc` with `cnn` and remove `--binary`). The datasets are downloaded automatically to the respective directories.*
