import math
import os
import pickle
import random
import time
from functools import cached_property

import numpy as np
import torch
import torch.nn.functional as F

from snn.core import get_device, package_path
from snn.core.extra_fn import (
    approximate_BPAC_bound,
    generate_noise,
    generate_zero_noise,
    next_batch,
    shuffledata,
)
from snn.core.mlp_fn import l2_norm
from snn.core.optim import TFRMSprop
from snn.core.sgd import SGD


def correct_predictions(yhat, y):
    """Boolean tensor of correct predictions. Binary models output a single logit and use labels in {-1, +1},
    multi-class models use one-hot labels."""
    if y.shape[-1] == 1:
        pred = (yhat >= 0).float() - (yhat < 0).float()
        return pred == y
    return torch.argmax(yhat, dim=1) == torch.argmax(y, dim=1)


class Network(object):
    """A network model.

    Subclasses (FC, CNN) have to provide `self.params` (list of leaf tensors holding the mean weights, in the same
    order and layout as the original TensorFlow implementation), `self.model(x, params)` (the forward pass),
    and `self.model_with_noise(x, noise_list, params)`.
    """

    def __init__(
        self,
        X,
        Y,
        logging=True,
        layers=[784, 600, 10],
        scopes_list=["hidden1", "output"],
        seed=11,
        laplace=False,
        device=None,
    ):
        self.device = get_device(device)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Storing variables
        self.X = X
        self.Y = Y
        self.layers = layers
        self.scopes_list = scopes_list
        self.Nsamples = self.X.shape[0]
        self.output_dict = {
            "L2": [],
            "diff": [],
            "weights": [],
            "mean_weights": [],
            "var_weights": [],
            "PACBound": [],
            "B_val": [],
            "KL_val": [],
            "test_acc": [],
            "train_acc": [],
            "L2_PACB": [],
            "log_post_all": [],
            "PACB_weights": [],
            "log_prior_std": [],
        }
        self.history = []

        # PACBound parameters
        self.log_prior_std_precision = 100.0
        self.log_prior_std_base = 0.1
        self.deltaPAC = 0.025

        # Set random seed
        self.seed = seed
        self.laplace = laplace
        self.params = None
        self._cost_described = False
        return

    def __call__(self, x, y):
        # Runs the accuracy of model on X
        self.print_accuracy(x, y)
        return

    def __del__(self):
        return

    # ------------------------------------------------------------------ helpers
    def _t(self, a):
        return torch.as_tensor(np.asarray(a), dtype=torch.float32, device=self.device)

    def _set_params(self, weights, requires_grad=True):
        self.params = [
            self._t(w).clone().requires_grad_(requires_grad) for w in weights
        ]

    def get_model_weights(self):
        return [p.detach().cpu().numpy().copy() for p in self.params]

    def count_N_params(self):
        shapes = [list(p.shape) for p in self.params]
        return int(sum(int(np.prod(s)) for s in shapes)), shapes

    def create_placeholders(self):
        # Kept for API compatibility: there are no placeholders in PyTorch
        return None, None

    # ------------------------------------------------------------------ accuracy
    def accuracy(self, x, y, params=None):
        """Mean accuracy of the (deterministic) model on a batch."""
        with torch.no_grad():
            yhat = self.model(self._t(x), self.params if params is None else params)
            return correct_predictions(yhat, self._t(y)).float().mean().item()

    def print_accuracy_in_batches(self, x, y, no_batches=10, whichset="train"):
        ntest, no_batches = x.shape[0], int(no_batches)
        testidx = np.linspace(0, ntest, no_batches + 1)
        test_acc = 0
        for ii, jj in zip(testidx[:-1], testidx[1:]):
            ii, jj = int(ii), int(jj)
            test_acc += self.accuracy(x[ii:jj], y[ii:jj])
        test_acc = test_acc / no_batches
        print("Average %s accuracy: %.4f" % (whichset, test_acc))
        return test_acc

    def print_accuracy_in_batches_noise(
        self, x, y, noise, scale_list, no_batches=10, whichset="train"
    ):
        """Accuracy of the stochastic network whose weights are perturbed by scale_list * noise."""
        ntest, no_batches = x.shape[0], int(no_batches)
        testidx = np.linspace(0, ntest, no_batches + 1)
        perturb = [s * self._t(n) for s, n in zip(scale_list, noise)]
        test_acc = 0
        for ii, jj in zip(testidx[:-1], testidx[1:]):
            ii, jj = int(ii), int(jj)
            with torch.no_grad():
                yhat = self.model_with_noise(self._t(x[ii:jj]), perturb, self.params)
                test_acc += (
                    correct_predictions(yhat, self._t(y[ii:jj])).float().mean().item()
                )
        test_acc = test_acc / no_batches
        print("Average %s accuracy: %.4f" % (whichset, test_acc))
        return test_acc

    def print_full_accuracy(self, x=None, y=None):
        print("Train Accuracy:", self.accuracy(self.X, self.Y))
        if x is not None:
            print("Test Accuracy:", self.accuracy(x, y))
        return

    def print_accuracy(self, x=None, y=None):
        if x is not None:
            print("Accuracy:", self.accuracy(x, y))
        return

    def return_accuracy(self, x, y):
        return self.accuracy(x, y)

    # ------------------------------------------------------------------ losses
    def logistic_loss(self, yhat, y):
        switch = (y * yhat < -y * yhat).float()
        exponent = torch.exp((2 * switch - 1) * y * yhat)
        return torch.mean(torch.log(1 + exponent) - (switch * y * yhat)) / np.float32(
            np.log(2)
        )

    def cost_fn(self, yhat, y):
        num_classes = y.shape[-1]
        if not self._cost_described:
            print("Number of classes: ", num_classes)
            print(
                "Minimizing logistic loss"
                if num_classes == 1
                else "Minimizing sigmoid_cross_entropy_with_logits"
            )
            self._cost_described = True
        if num_classes == 1:
            return self.logistic_loss(yhat, y)
        else:
            return torch.mean(
                F.binary_cross_entropy_with_logits(yhat, y, reduction="none")
            ) / math.log(num_classes)

    # ------------------------------------------------------------------ SGD
    def optimize(self, epochs=10, batch_size=100, learning_rate=0.01, momentum=0.9):
        for p in self.params:
            p.requires_grad_(True)
        train_step = torch.optim.SGD(self.params, lr=learning_rate, momentum=momentum)
        weights_rand_init = self.get_model_weights()
        # Initialize the SGD trainer
        sgd = SGD(self.X, self.Y, total_epochs=epochs, batch_size=100, seed=self.seed)
        number_of_batches_per_epoch = int(self.Nsamples / batch_size)
        total_number_of_iterations = epochs * number_of_batches_per_epoch
        for i in range(total_number_of_iterations):
            epoch = int(i / number_of_batches_per_epoch)
            if i % 100 == 0:
                remainder = i % number_of_batches_per_epoch
                print("Iter: %d / %d" % (remainder, number_of_batches_per_epoch))

            if i % number_of_batches_per_epoch == 0:  # Print and shuffle at every epoch
                print("Epoch: ", epoch)
                self.print_accuracy_in_batches(self.X, self.Y, 50)
                index_set = sgd.epoch_train_set(epoch=epoch)

            b = int(i % number_of_batches_per_epoch)
            batch_idx = index_set[b * batch_size : (b + 1) * batch_size]
            batch_x, batch_y = self._t(self.X[batch_idx]), self._t(self.Y[batch_idx])
            cost = self.cost_fn(self.model(batch_x, self.params), batch_y)
            train_step.zero_grad()
            cost.backward()
            train_step.step()
        return weights_rand_init

    def save_output(
        self, path=os.path.join(package_path, "experiments", "results", "out.pickle")
    ):
        save_dir = os.path.dirname(path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(path, "wb") as f:
            pickle.dump(self.output_dict, f)
        return

    def save_model(self, save_tag):
        return

    def log_res(self, res, log_tag):
        """Logs the result"""
        return

    # ------------------------------------------------------------------ PAC-Bayes
    @cached_property  # This causes PACB_init to only be computed once
    def PACB_init(self):
        """Initializes variables for the PAC Bound optimization"""
        network_weights = self.get_model_weights()

        # The initial standard deviations
        init_log_prior_std = -3.0
        log_post_std_list = []
        for _w in network_weights:
            log_post_std_init = np.log(2 * np.abs(_w))
            log_post_std_list.append(self._t(log_post_std_init).requires_grad_(True))

        log_prior_std = torch.full(
            [1],
            init_log_prior_std,
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        return network_weights, log_post_std_list, log_prior_std

    def _perturbations(self, log_post_std_list, noise_list):
        return [
            torch.exp(log_post_std) * self._t(noise)
            for log_post_std, noise in zip(log_post_std_list, noise_list)
        ]

    def PACB_objective(self, prior_weights, trainWeights=True):
        """Prepares the optimization of the PAC-Bayes bound and returns a function evaluating it on a batch.
        The returned function maps (x, y, noise_list) to
        (A, cost, Bquad, effective_m, log_prior_std, log_post_std_list, factor1, factor2)."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        # Obtain the effective number of data points
        effective_m = self.X.shape[0]

        # Gather the initial weights to create a new optimization network
        network_weights, log_post_std_list, log_prior_std = self.PACB_init

        # The means of the stochastic network are initialised at the weights found by SGD
        self._set_params(network_weights, requires_grad=trainWeights)

        nparams, layer_shapes = self.count_N_params()
        self.layer_shapes = layer_shapes
        norm_params = self._t(
            sum(np.sum((x - y) ** 2) for x, y in zip(network_weights, prior_weights))
        )
        factor1 = 2 * math.log(self.log_prior_std_precision)

        def objective(x, y, noise_list):
            network_perturb_list = self._perturbations(log_post_std_list, noise_list)
            self.yhat = self.model_with_noise(x, network_perturb_list, self.params)

            A = self.cost_fn(self.yhat, y)

            if self.laplace:
                b0 = log_prior_std
                KL_div = 0.0
                mean_diff_sum = 0.0

                for p, p0_np, log_post_std in zip(
                    self.params, prior_weights, log_post_std_list
                ):
                    p0 = self._t(p0_np)
                    b1 = torch.exp(log_post_std)
                    diff_abs = torch.abs(p - p0)

                    term1 = torch.log(b0) - log_post_std
                    term2 = diff_abs / b0
                    term3 = (b1 / b0) * torch.exp(-diff_abs / b1)
                    term4 = -1.0

                    KL_div += torch.sum(term1 + term2 + term3 + term4)
                    mean_diff_sum += torch.sum(term2)

                self.KLdivTimes2 = 2.0 * KL_div
                self.mean_weights_component = mean_diff_sum
                self.var_weights_component = KL_div - mean_diff_sum
            else:
                norm_post_variance = sum(
                    torch.sum(torch.exp(s * 2)) for s in log_post_std_list
                )
                sum_log_post_variance = sum(torch.sum(s) for s in log_post_std_list)

                self.mean_weights_component = (norm_params) / (
                    (self.log_prior_std_base - 0.001) / (1 + torch.exp(-log_prior_std))
                )
                self.var_weights_component = (
                    norm_post_variance
                    / (
                        (self.log_prior_std_base - 0.001)
                        / (1 + torch.exp(-log_prior_std))
                    )
                    - 2 * sum_log_post_variance
                    + nparams
                    * torch.log(
                        (self.log_prior_std_base - 0.001)
                        / (1 + torch.exp(-log_prior_std))
                    )
                )
                self.KLdivTimes2 = (
                    self.mean_weights_component + self.var_weights_component - nparams
                )

            f1 = torch.tensor(factor1, device=self.device)
            factor2 = 2 * torch.log(
                math.log(self.log_prior_std_base)
                - torch.log(
                    (self.log_prior_std_base - 0.001) / (1 + torch.exp(-log_prior_std))
                )
            )
            Bquad = (
                self.KLdivTimes2 / 2
                + math.log(np.pi**2 * effective_m / (6 * 0.05))
                + f1
                + factor2
            )

            c = Bquad / (2 * (effective_m - 1))
            self.B = torch.sqrt(c)
            cost = A + self.B
            return (
                A,
                cost,
                Bquad,
                effective_m,
                log_prior_std,
                log_post_std_list,
                f1,
                factor2,
            )

        return objective

    def PACB_store(
        self,
        save_dict,
        i,
        log_prior_std=None,
        m_w=None,
        v_w=None,
        bpac=None,
        B_val=None,
        KL_val=None,
        test_acc=None,
        train_acc=None,
        log_post_std_list=None,
    ):
        if save_dict is not None:
            if (
                i % save_dict["iter"] == 0
            ):  # Only save at a certain number of iterations
                if save_dict["mean_weights"] and m_w is not None:
                    self.output_dict["mean_weights"].append(m_w)
                if save_dict["var_weights"] and v_w is not None:
                    self.output_dict["var_weights"].append(v_w)
                if save_dict["PACBound"] and bpac is not None:
                    self.output_dict["PACBound"].append(
                        bpac
                    )  # Store the desired values for saving
                if save_dict["B_val"] and B_val is not None:
                    self.output_dict["B_val"].append(
                        B_val
                    )  # Store the desired values for saving
                if save_dict["KL_val"] and KL_val is not None:
                    self.output_dict["KL_val"].append(
                        KL_val
                    )  # Store the desired values for saving
                if save_dict["test_acc"] and test_acc is not None:
                    self.output_dict["test_acc"].append(
                        test_acc
                    )  # Store the desired values for saving
                if save_dict["train_acc"] and train_acc is not None:
                    self.output_dict["train_acc"].append(
                        train_acc
                    )  # Store the desired values for saving
                if save_dict["L2_PACB"]:
                    self.output_dict["L2_PACB"].append(
                        l2_norm(self.get_model_weights(), save_dict["w*"])
                    )
                if save_dict["log_post_all"] and log_post_std_list is not None:
                    noisevecs = []
                    for log_post_std in log_post_std_list:
                        noisevecs.append(log_post_std.detach().cpu().numpy())
                    log_post_std_all = []
                    for nv in noisevecs:
                        for v in nv.flatten():
                            log_post_std_all.append(v)
                    self.output_dict["log_post_all"].append(log_post_std_all)
                if save_dict["PACB_weights"]:
                    self.output_dict["PACB_weights"].append(self.get_model_weights())
                if save_dict["log_prior_std"]:
                    self.output_dict["log_prior_std"].append(log_prior_std)
        return

    def _pacb_correct_sum(self, bx, by, log_post_std_list, noise_list):
        with torch.no_grad():
            perturb = self._perturbations(log_post_std_list, noise_list)
            yhat = self.model_with_noise(self._t(bx), perturb, self.params)
            return correct_predictions(yhat, self._t(by)).float().sum().item()

    def optimize_PACB(
        self,
        prior_weights,
        epochs=20,
        learning_rate=0.01,
        batch_size=100,
        drop_lr=10,
        lr_factor=0.05,
        save_dict=None,
        trainWeights=True,
    ):
        """Optimize the PAC Bayes Bound depending on a prior"""

        objective = self.PACB_objective(prior_weights, trainWeights=trainWeights)
        _, log_post_std_list, log_prior_std = self.PACB_init
        trainable = (
            [log_prior_std]
            + list(log_post_std_list)
            + (list(self.params) if trainWeights else [])
        )
        train_step = TFRMSprop(trainable, lr=learning_rate)
        lr_dropped = False

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        mean_accuracy_stoch_print = []
        mean_accuracy_det_print = []
        self.mean_weights_list = []
        self.var_weights_list = []
        self.pacb_list = []
        Nsamples = self.Nsamples
        trainX, trainY = self.X, self.Y
        bpac = None

        for i in range(epochs * int(Nsamples / batch_size)):
            epoch = int(i / (Nsamples / batch_size))
            batch_x, batch_y = next_batch(
                trainX, trainY, batch_size, int(i % (Nsamples / batch_size))
            )

            # For the stochastic network
            noise_list = generate_noise(self.layer_shapes, self.laplace)

            if i % int(Nsamples / batch_size) == 0:  # At every epoch, shuffle the data
                trainX, trainY = shuffledata(trainX, trainY)

                train_accuracy_stoch = 0
                train_accuracy_det = 0
                for ib in range(int(Nsamples / batch_size)):
                    bx, by = next_batch(
                        trainX, trainY, batch_size, int(ib % (Nsamples / batch_size))
                    )

                    # Find accuracy of the stochastic network
                    noise_list = generate_noise(self.layer_shapes, self.laplace)
                    train_accuracy_stoch += self._pacb_correct_sum(
                        bx, by, log_post_std_list, noise_list
                    )

                    # Find accuracy of the deterministic network
                    zero_noise_list = generate_zero_noise(self.layer_shapes)
                    train_accuracy_det += self._pacb_correct_sum(
                        bx, by, log_post_std_list, zero_noise_list
                    )

                train_accuracy_stoch = train_accuracy_stoch / Nsamples
                train_accuracy_det = train_accuracy_det / Nsamples
                mean_accuracy_stoch_print.append(train_accuracy_stoch)
                mean_accuracy_det_print.append(train_accuracy_det)

            if (drop_lr is not None) and (epoch > drop_lr) and not lr_dropped:
                train_step = TFRMSprop(trainable, lr=lr_factor * learning_rate)
                lr_dropped = True

            A, cost, Bquad, _, _, _, factor1, factor2 = objective(
                self._t(batch_x), self._t(batch_y), noise_list
            )
            train_step.zero_grad()
            cost.sum().backward()
            train_step.step()

            A_i, cost_i, kldiv2_i, B_i = (
                A.item(),
                cost.item(),
                self.KLdivTimes2.item(),
                self.B.item(),
            )
            m_w, v_w = (
                self.mean_weights_component.item(),
                self.var_weights_component.item(),
            )
            _log_prior_std, Bquad_i = log_prior_std.item(), Bquad.item()
            factor1_i, factor2_i = factor1.item(), factor2.item()

            if i % (1 * Nsamples / batch_size) == 0 or (
                i == epochs * int(Nsamples / batch_size) - 1
            ):
                bpac = approximate_BPAC_bound(train_accuracy_stoch, B_i)
                self.history.append(
                    {
                        "epoch": epoch,
                        "A_term": A_i,
                        "B_term": B_i,
                        "objective": cost_i,
                        "stochastic_train_accuracy": train_accuracy_stoch,
                        "stochastic_train_error": 1.0 - train_accuracy_stoch,
                        "KL": kldiv2_i / 2,
                        "PAC_bound_estimate": bpac,
                        "log_prior_std": _log_prior_std,
                        "learning_rate": lr_factor * learning_rate
                        if lr_dropped
                        else learning_rate,
                    }
                )
            if i % (1 * Nsamples / batch_size) == 0 or (
                i == epochs * int(Nsamples / batch_size) - 1
            ):
                bpac = approximate_BPAC_bound(train_accuracy_stoch, B_i)
                output = "".join(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                    + "Epoch:"
                    + "%04d" % (epoch + 1)
                    + " cost="
                    + str(cost_i)
                    + " mean accuracy %.4f" % train_accuracy_stoch
                    + " KL div:  %.4f" % (kldiv2_i / 2)
                    + " A term: %.4f" % A_i
                    + " B term: %.4f" % B_i
                    + " Bquad: %.4f" % Bquad_i
                    + " log_prior_std: %.4f" % _log_prior_std
                    + " B PAC: %.4f" % bpac
                    + " factor1: %.4f" % factor1_i
                    + " factor2: %.4f" % factor2_i
                )
                print(output)
            self.PACB_store(
                save_dict,
                i=i,
                log_prior_std=np.array([_log_prior_std], dtype=np.float32),
                m_w=m_w,
                v_w=v_w,
                bpac=bpac,
                log_post_std_list=log_post_std_list,
                KL_val=kldiv2_i / 2,
            )  # Store the desired values for saving

        # Save log_prior_std and log_posterior for evaluate_SNN_accuracy()
        self.log_prior_std = log_prior_std.detach().cpu().numpy()
        noisevecs = []
        for log_post_std in log_post_std_list:
            noisevecs.append(log_post_std.detach().cpu().numpy())
        self.log_post_all = []
        for nv in noisevecs:
            for v in nv.flatten():
                self.log_post_all.append(v)

        return

    def evaluate_SNN_accuracy(
        self, testX, testY, prior_weights=None, N_SNN_samples=200, save_dict=None
    ):
        """Run the check accuracy code"""

        params_mean_values = self.get_model_weights()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        nparams, parameter_shapes = self.count_N_params()
        effective_m = self.X.shape[0]

        params_means = []
        for a, b in zip(params_mean_values, prior_weights):
            params_means.append(a - b)

        log_post_all = self.log_post_all
        log_post_std_init_list = []
        for par_shape, par_mean in zip(self.layer_shapes, params_means):
            log_post_std_init_tmp = np.reshape(
                log_post_all[: int(np.prod(par_shape))], par_shape
            )
            assert log_post_std_init_tmp.shape == par_mean.shape
            log_post_std_init_list.append(log_post_std_init_tmp)
            log_post_all = log_post_all[int(np.prod(par_shape)) :]

        # Load and discretize prior variance parameter
        init_log_prior_std = float(np.asarray(self.log_prior_std).reshape(-1)[0])
        jdisc = self.log_prior_std_precision * (
            np.log(self.log_prior_std_base)
            - np.log(
                (self.log_prior_std_base - 0.001) / (1 + np.exp(-init_log_prior_std))
            )
        )
        print("Before discretization")
        print(init_log_prior_std, jdisc)
        jdisc_up = np.maximum(np.float32(math.ceil(jdisc)), 1)
        jdisc_down = np.maximum(np.float32(math.floor(jdisc)), 1)
        init_log_prior_std_up = np.exp(
            np.log(self.log_prior_std_base) - jdisc_up / self.log_prior_std_precision
        )
        init_log_prior_std_down = np.exp(
            np.log(self.log_prior_std_base) - jdisc_down / self.log_prior_std_precision
        )
        print("After discretization")
        print(init_log_prior_std_down, jdisc_down)
        print(init_log_prior_std_up, jdisc_up)

        log_post_std_list = [
            np.asarray(v, dtype=np.float64) for v in log_post_std_init_list
        ]
        scale_list = [self._t(np.exp(v)) for v in log_post_std_list]

        if self.laplace:

            def KLdivTimes2(prior_scale):
                b0 = prior_scale
                kl = 0.0
                for diff, log_b1 in zip(params_means, log_post_std_list):
                    b1 = np.exp(log_b1)
                    abs_diff = np.abs(diff)

                    term1 = np.log(b0) - log_b1
                    term2 = abs_diff / b0
                    term3 = (b1 / b0) * np.exp(-abs_diff / b1)
                    term4 = -1.0

                    kl += np.sum(term1 + term2 + term3 + term4)
                return 2.0 * kl
        else:
            norm_post_variance = sum(np.sum(np.exp(x * 2)) for x in log_post_std_list)
            norm_params = sum(
                np.sum(np.asarray(x, dtype=np.float64) ** 2) for x in params_means
            )
            sum_log_post_variance = sum(np.sum(x) for x in log_post_std_list)

            def KLdivTimes2(log_prior_std):
                return (
                    (norm_post_variance + norm_params) / log_prior_std
                    + nparams * math.log(log_prior_std)
                    - nparams
                    - 2 * sum_log_post_variance
                )

        def B_fn(log_prior_std, jopt):
            Bquad = (
                KLdivTimes2(log_prior_std) / 2
                + np.log(np.pi**2 * effective_m / (6 * self.deltaPAC))
                + 2 * np.log(jopt)
            )
            return np.sqrt(Bquad / (2 * (effective_m - 1)))

        mean_train_accuracy = 0
        mean_test_accuracy = 0
        for ns in range(N_SNN_samples):
            noise_list = generate_noise(self.layer_shapes, self.laplace)

            train_accur_i = self.print_accuracy_in_batches_noise(
                self.X, self.Y, noise_list, scale_list, whichset="train"
            )

            test_accur_i = self.print_accuracy_in_batches_noise(
                testX, testY, noise_list, scale_list, whichset="test"
            )
            mean_train_accuracy += train_accur_i
            mean_test_accuracy += test_accur_i

        mean_train_accuracy = mean_train_accuracy / N_SNN_samples
        mean_test_accuracy = mean_test_accuracy / N_SNN_samples
        print(
            "Train error :",
            (1 - mean_train_accuracy),
            "Test error :",
            (1 - mean_test_accuracy),
        )

        B_valD = B_fn(init_log_prior_std_down, jdisc_down)
        B_valU = B_fn(init_log_prior_std_up, jdisc_up)
        B_val = np.minimum(B_valU, B_valD)
        if B_valU < B_valD:
            KL_val = KLdivTimes2(init_log_prior_std_up) / 2.0
        else:
            KL_val = KLdivTimes2(init_log_prior_std_down) / 2.0

        bpac = approximate_BPAC_bound(mean_train_accuracy, B_val)
        print("Results with delta = %.3f" % (self.deltaPAC + 0.01))
        print(
            "PAC bound error:",
            "%.4f" % bpac,
            "Gen bound :",
            "%.4f" % B_val,
            "KL value: ",
            "%.4f" % KL_val,
        )

        self.PACB_store(
            save_dict,
            i=0,
            log_prior_std=self.log_prior_std,
            bpac=bpac,
            B_val=B_val,
            KL_val=KL_val,
            test_acc=mean_test_accuracy,
            train_acc=mean_train_accuracy,
        )
        return (
            bpac,
            B_val,
            KL_val,
            self.deltaPAC,
            mean_train_accuracy,
            mean_test_accuracy,
            self.log_prior_std,
        )
