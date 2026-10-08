# Changes from the upstream TensorFlow implementation are documented beside the
# affected operations below. The PyTorch port keeps the original parameter order
# and Gaussian SNN family, while correcting the live-mean KL term, prior grid,
# final Monte Carlo certificate, and saved posterior consistency.
import os
import math
import pickle
import random
from functools import cached_property

import numpy as np
import torch
import torch.nn.functional as F

from snn.core.extra_fn import (generate_noise, generate_zero_noise, approximate_BPAC_bound, next_batch,
                               shuffledata, discretized_prior_bound, inverse_binary_kl,
                               monte_carlo_error_upper)
from snn.core.mlp_fn import l2_norm
from snn.core.optim import TFRMSprop
from snn.core.sgd import SGD
from snn.core import package_path, get_device


def correct_predictions(yhat, y):
    """ Boolean tensor of correct predictions. Binary models output a single logit and use labels in {-1, +1},
    multi-class models use one-hot labels. """
    if y.shape[-1] == 1:
        pred = (yhat >= 0).float() - (yhat < 0).float()
        return pred == y
    return torch.argmax(yhat, dim=1) == torch.argmax(y, dim=1)


class Network(object):
    """ A network model.

    Subclasses (FC, CNN) have to provide `self.params` (list of leaf tensors holding the mean weights, in the same
    order and layout as the original TensorFlow implementation), `self.model(x, params)` (the forward pass),
    and `self.model_with_noise(x, noise_list, params)`.
    """

    def __init__(self, X, Y, logging=True, layers=[784, 600, 10], scopes_list=['hidden1', 'output'], seed=11,
                 device=None):
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
        self.output_dict = {"L2": [], "diff": [], "weights": [], "mean_weights": [], "var_weights": [], "PACBound": [], "B_val": [], "KL_val": [], "test_acc": [], "train_acc": [], "L2_PACB": [], "log_post_all": [], "PACB_weights": [], "log_prior_std": []}
        self.history = []

        # PACBound parameters
        self.log_prior_std_precision = 100.0
        self.log_prior_std_base = 0.1
        # The paper uses delta_PAC = 0.025 and delta_MC = 0.01. The upstream
        # training objective instead hardcoded 0.05, and its final evaluator
        # printed a combined delta without applying the MC correction.
        # Here training and final PAC calculations share deltaPAC; final MC
        # evaluation adds deltaMC, giving confidence 1 - 0.025 - 0.01 = 0.965
        # for each experiment under the corresponding theorem assumptions.
        self.deltaPAC = 0.025
        self.deltaMC = 0.01

        # Set random seed
        self.seed = seed
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
        # A NumPy weight snapshot is only an initializer, not a live reference.
        # These leaf tensors become the current posterior means. Prediction and
        # the KL objective must both read self.params after optimizer updates.
        # requires_grad=False freezes every mean weight and bias, but does not
        # freeze the separately created posterior or prior log standard deviations.
        self.params = [self._t(w).clone().requires_grad_(requires_grad) for w in weights]

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
        """ Mean accuracy of the (deterministic) model on a batch. """
        with torch.no_grad():
            yhat = self.model(self._t(x), self.params if params is None else params)
            return correct_predictions(yhat, self._t(y)).float().mean().item()

    def print_accuracy_in_batches(self, x, y, no_batches=10, whichset='train'):
        ntest, no_batches = x.shape[0], int(no_batches)
        testidx = np.linspace(0, ntest, no_batches+1)
        test_acc = 0
        for (ii, jj) in zip(testidx[:-1], testidx[1:]):
            ii, jj = int(ii), int(jj)
            test_acc += self.accuracy(x[ii:jj], y[ii:jj])
        test_acc = test_acc / no_batches
        print("Average %s accuracy: %.4f" %(whichset, test_acc))
        return test_acc

    def print_accuracy_in_batches_noise(self, x, y, noise, scale_list, no_batches=10, whichset='train', verbose=True):
        """ Accuracy of the stochastic network whose weights are perturbed by scale_list * noise. """
        ntest, no_batches = x.shape[0], min(x.shape[0], int(no_batches))
        if ntest < 1 or no_batches < 1:
            raise ValueError("Accuracy evaluation requires nonempty data and positive batch count")
        testidx = np.linspace(0, ntest, no_batches + 1)
        # Keep one sampled network fixed across all data batches. Splitting the
        # data is a memory optimization, not an additional source of MC samples.
        perturb = [s * self._t(n) for s, n in zip(scale_list, noise)]
        test_acc = 0
        for (ii, jj) in zip(testidx[:-1], testidx[1:]):
            ii, jj = int(ii), int(jj)
            with torch.no_grad():
                yhat = self.model_with_noise(self._t(x[ii:jj]), perturb, self.params)
                # Count correct examples before dividing by the full set size.
                # Averaging batch accuracies would misweight unequal batches;
                # the upstream equal-batch average only agrees for equal sizes.
                test_acc += correct_predictions(yhat, self._t(y[ii:jj])).sum().item()
        test_acc = test_acc / ntest
        if verbose:
            print("Average %s accuracy: %.4f" %(whichset, test_acc))
        return test_acc

    def print_full_accuracy(self, x=None, y=None):
        print("Train Accuracy:", self.accuracy(self.X, self.Y))
        if x is not None: print("Test Accuracy:", self.accuracy(x, y))
        return

    def print_accuracy(self, x=None, y=None):
        if x is not None: print("Accuracy:", self.accuracy(x, y))
        return

    def return_accuracy(self, x, y):
        return self.accuracy(x, y)

    # ------------------------------------------------------------------ losses
    def logistic_loss(self, yhat, y):
        # softplus(z) is algebraically log(1 + exp(z)), but remains finite for
        # large logits where the upstream explicit exponential can overflow.
        # Division by log(2) preserves the paper's logistic surrogate scaling.
        return F.softplus(-y * yhat).mean() / math.log(2)

    def cost_fn(self, yhat, y):
        num_classes = y.shape[-1]
        if not self._cost_described:
            print("Number of classes: ", num_classes)
            print("Minimizing logistic loss" if num_classes == 1 else "Minimizing sigmoid_cross_entropy_with_logits")
            self._cost_described = True
        if num_classes == 1:
            return self.logistic_loss(yhat, y)
        else:
            return (torch.mean(F.binary_cross_entropy_with_logits(yhat, y, reduction="none")) /
                    math.log(num_classes))

    # ------------------------------------------------------------------ SGD
    def optimize(self, epochs=10, batch_size=100, learning_rate=0.01, momentum=0.9, max_steps=None):
        # Added for the initialization ablations: max_steps=1 means exactly one
        # mini-batch SGD update, not one epoch (550 updates on binary MNIST).
        # Zero epochs preserve fresh random weights. With no step limit, the
        # baseline and frozen-mean condition retain their full SGD epoch budget.
        if max_steps is not None and max_steps < 0:
            raise ValueError("max_steps must be nonnegative")
        for p in self.params:
            p.requires_grad_(True)
        train_step = torch.optim.SGD(self.params, lr=learning_rate, momentum=momentum)
        # Capture the random initialization before any SGD step. It remains the
        # fixed prior mean w0 during the subsequent PAC-Bayes optimization;
        # the SGD solution initializes the posterior mean, not the prior mean.
        weights_rand_init = self.get_model_weights()
        # Initialize the SGD trainer
        sgd = SGD(self.X, self.Y, total_epochs=epochs, batch_size=100, seed=self.seed)
        number_of_batches_per_epoch = int(self.Nsamples / batch_size)
        total_number_of_iterations = epochs * number_of_batches_per_epoch
        if max_steps is not None:
            total_number_of_iterations = min(total_number_of_iterations, max_steps)
        self.sgd_steps_completed = 0
        for i in range(total_number_of_iterations):
            epoch = int(i / number_of_batches_per_epoch)
            if i%100==0:
              remainder = i % number_of_batches_per_epoch
              print("Iter: %d / %d"%(remainder, number_of_batches_per_epoch)
                    + f" SGD iteration: {i}/{total_number_of_iterations}")

            if i%number_of_batches_per_epoch == 0: # Print and shuffle at every epoch
                print("Epoch: ", epoch)
                self.print_accuracy_in_batches(self.X, self.Y,50)
                index_set = sgd.epoch_train_set(epoch=epoch)

            b = int(i % number_of_batches_per_epoch)
            batch_idx = index_set[b * batch_size:(b + 1) * batch_size]
            batch_x, batch_y = self._t(self.X[batch_idx]), self._t(self.Y[batch_idx])
            cost = self.cost_fn(self.model(batch_x, self.params), batch_y)
            train_step.zero_grad()
            cost.backward()
            train_step.step()
            self.sgd_steps_completed += 1
        return weights_rand_init

    def save_output(self, path=os.path.join(package_path, "experiments", "results", "out.pickle")):
        save_dir = os.path.dirname(path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(path, 'wb') as f:
            pickle.dump(self.output_dict, f)
        return

    def save_model(self, save_tag):
        return

    def log_res(self, res, log_tag):
        """ Logs the result """
        return

    # ------------------------------------------------------------------ PAC-Bayes
    @cached_property # This causes PACB_init to only be computed once
    def PACB_init(self):
        """ Initializes variables for the PAC Bound optimization """
        # Like upstream, this cached property takes a one-time snapshot at the
        # start of stage two. It must not supply the mean distance throughout
        # training: network_weights does not change when self.params changes.
        network_weights = self.get_model_weights()

        # The initial standard deviations
        init_log_prior_std = -3.0
        log_post_std_list = []
        for _w in network_weights:
            # Upstream initializes std = 2 * abs(weight), which gives log(0)
            # for zero biases. Floor only the standard deviation to 1e-8; do
            # not perturb the mean or prior to hide this numerical problem.
            log_post_std_init = np.log(np.maximum(2 * np.abs(_w), np.float32(1e-8)))
            log_post_std_list.append(self._t(log_post_std_init).requires_grad_(True))

        log_prior_std = torch.full([1], init_log_prior_std, dtype=torch.float32, device=self.device,
                                   requires_grad=True)
        return network_weights, log_post_std_list, log_prior_std

    def _perturbations(self, log_post_std_list, noise_list):
        return [torch.exp(log_post_std) * self._t(noise) for log_post_std, noise in zip(log_post_std_list, noise_list)]

    def PACB_objective(self, prior_weights, trainWeights=True):
        """ Prepares the optimization of the PAC-Bayes bound and returns a function evaluating it on a batch.
        The returned function maps (x, y, noise_list) to
        (A, cost, Bquad, effective_m, log_prior_std, log_post_std_list, factor1, factor2). """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        # Obtain the effective number of data points
        effective_m = self.X.shape[0]

        # Gather the initial weights to create a new optimization network
        network_weights, log_post_std_list, log_prior_std = self.PACB_init

        # Initialize live posterior means from the chosen stage-one result:
        # random weights, a short SGD warmup, or the full SGD solution.
        # In no_trainw, trainWeights=False fixes the full SGD mean. Samples still
        # vary because the posterior standard deviations remain trainable.
        self._set_params(network_weights, requires_grad=trainWeights)

        nparams, layer_shapes = self.count_N_params()
        self.layer_shapes = layer_shapes
        # The prior mean must stay fixed at the pre-SGD random initialization.
        # Clone it separately so updates to the posterior cannot change it.
        prior_tensors = [self._t(w).clone() for w in prior_weights]
        if len(prior_tensors) != len(self.params) or any(
                w.shape != w0.shape for w, w0 in zip(self.params, prior_tensors)):
            raise ValueError("Prior parameters must match the posterior parameter shapes")
        if effective_m <= 1:
            raise ValueError("PAC-Bayes optimization requires at least two training examples")
        factor1 = 2 * math.log(self.log_prior_std_precision)

        def objective(x, y, noise_list):
            network_perturb_list = self._perturbations(log_post_std_list, noise_list)
            self.yhat = self.model_with_noise(x, network_perturb_list, self.params)

            # Main correction to upstream network.py:228: upstream computes
            # this distance from cached NumPy network_weights, even though its
            # forward pass trains newly created param_var_list variables.
            # Using the live self.params here, inside each objective call,
            # makes prediction and KL describe the same current posterior Q.
            # For lambda = exp(2 * log_prior_std), this contributes
            # ||mu - w0||^2 / (2 * lambda) to KL, with mean gradient
            # (mu - w0) / lambda. That gradient was missing upstream.
            # In the frozen-mean ablation this distance intentionally stays
            # constant; the full KL can still change with posterior/prior variance.
            norm_params = sum(torch.sum((w - w0) ** 2) for w, w0 in zip(self.params, prior_tensors))
            # Let r = log(sigma_i^2 / lambda). Then expm1(r) - r equals
            # sigma_i^2 / lambda - 1 - log(sigma_i^2 / lambda), the original
            # Gaussian covariance term. expm1 avoids cancellation near r=0;
            # this changes numerical stability, not the mathematical Gaussian KL.
            covariance_component = sum(torch.sum(torch.expm1(2 * (s - log_prior_std))
                                                 - 2 * (s - log_prior_std))
                                       for s in log_post_std_list)

            A = self.cost_fn(self.yhat, y)

            self.mean_weights_component = (norm_params) / (torch.exp(2 * log_prior_std))
            self.var_weights_component = covariance_component + nparams
            # Store twice the Gaussian KL for compatibility with the original
            # logging convention. Divide by two when reporting KL itself.
            self.KLdivTimes2 = self.mean_weights_component + covariance_component
            f1 = log_prior_std.new_tensor(factor1)
            factor2 = 2 * torch.log(torch.clamp(math.log(self.log_prior_std_base) - 2 * log_prior_std,
                                               min=1 / self.log_prior_std_precision))
            Bquad = self.KLdivTimes2 / 2 + math.log(np.pi ** 2 * effective_m / (6 * self.deltaPAC)) + f1 + factor2

            c = Bquad / (2 * (effective_m - 1))
            self.B = torch.sqrt(c)
            cost = A + self.B
            return A, cost, Bquad, effective_m, log_prior_std, log_post_std_list, f1, factor2

        return objective

    def PACB_store(self, save_dict, i, log_prior_std=None, m_w=None, v_w=None, bpac=None, B_val=None, KL_val=None,
                   test_acc=None, train_acc=None, log_post_std_list=None):
        if save_dict is not None:
            if i % save_dict["iter"] == 0: # Only save at a certain number of iterations
                if save_dict["mean_weights"] and m_w is not None:
                    self.output_dict["mean_weights"].append(m_w)
                if save_dict["var_weights"] and v_w is not None:
                    self.output_dict["var_weights"].append(v_w)
                if save_dict["PACBound"] and bpac is not None:
                    self.output_dict["PACBound"].append(bpac) # Store the desired values for saving
                if save_dict["B_val"] and B_val is not None:
                    self.output_dict["B_val"].append(B_val) # Store the desired values for saving
                if save_dict["KL_val"] and KL_val is not None:
                    self.output_dict["KL_val"].append(KL_val) # Store the desired values for saving
                if save_dict["test_acc"] and test_acc is not None:
                    self.output_dict["test_acc"].append(test_acc) # Store the desired values for saving
                if save_dict["train_acc"] and train_acc is not None:
                    self.output_dict["train_acc"].append(train_acc) # Store the desired values for saving
                if save_dict["L2_PACB"]:
                    self.output_dict["L2_PACB"].append(l2_norm(self.get_model_weights(), save_dict["w*"]))
                if save_dict["log_post_all"] and log_post_std_list is not None:
                    noisevecs = []
                    for log_post_std in log_post_std_list:
                        noisevecs.append(log_post_std.detach().cpu().numpy().copy())
                    log_post_std_all = []
                    for nv in noisevecs:
                        for v in nv.flatten():
                              log_post_std_all.append(v)
                    self.output_dict["log_post_all"].append(  log_post_std_all)
                if save_dict["PACB_weights"]:
                    self.output_dict["PACB_weights"].append(self.get_model_weights())
                if save_dict["log_prior_std"]:
                    self.output_dict["log_prior_std"].append(log_prior_std)
        return

    def _project_prior_std(self, log_prior_std):
        """Project onto lambda <= c exp(-1/b), so every grid index is positive."""
        # The union bound is over positive integer j with lambda_j=c*exp(-j/b).
        # A continuous prior outside this domain has no valid neighboring grid
        # certificate. Project after each step instead of repairing an invalid
        # index with abs(j), which would refer to a different prior variance.
        upper = (math.log(self.log_prior_std_base) - 1 / self.log_prior_std_precision) / 2
        # Round downward to keep the float32 boundary inside the feasible set.
        upper = float(np.nextafter(np.float32(upper), np.float32(-np.inf)))
        with torch.no_grad():
            log_prior_std.clamp_(max=upper)

    def _pacb_correct_sum(self, bx, by, log_post_std_list, noise_list):
        with torch.no_grad():
            perturb = self._perturbations(log_post_std_list, noise_list)
            yhat = self.model_with_noise(self._t(bx), perturb, self.params)
            return correct_predictions(yhat, self._t(by)).float().sum().item()

    def optimize_PACB(self, prior_weights, epochs=20, learning_rate=0.01, batch_size=100, drop_lr=10, lr_factor=0.05,
                      save_dict=None, trainWeights=True, eval_interval=1):
        """ Optimize the PAC Bayes Bound depending on a prior """

        if eval_interval < 0:
            raise ValueError("eval_interval must be nonnegative")

        objective = self.PACB_objective(prior_weights, trainWeights=trainWeights)
        _, log_post_std_list, log_prior_std = self.PACB_init
        self._project_prior_std(log_prior_std)
        # The trainWeights switch controls only posterior means. Explicitly
        # exclude them from RMSProp when False, as well as disabling their
        # gradients in _set_params. Posterior log std and prior log std remain
        # in the optimizer in every condition, so no_trainw still learns an SNN.
        trainable = [log_prior_std] + list(log_post_std_list) + (list(self.params) if trainWeights else [])
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
        train_accuracy_stoch = None
        batches_per_epoch = int(Nsamples / batch_size)
        total_iterations = epochs * batches_per_epoch

        for i in range(total_iterations):
            epoch = int(i / (Nsamples/batch_size))
            # Shuffle before taking the epoch's first mini-batch. Upstream took
            # that batch before shuffling, so its first update used the old order.
            if i % batches_per_epoch == 0: # At every epoch, shuffle the data
                trainX, trainY = shuffledata(trainX, trainY)
            batch_x, batch_y = next_batch(trainX, trainY, batch_size, i % batches_per_epoch)
            noise_list = generate_noise(self.layer_shapes)

            if (drop_lr is not None) and (epoch>drop_lr) and not lr_dropped:
                train_step = TFRMSprop(trainable, lr=lr_factor*learning_rate)
                lr_dropped = True

            A, cost, Bquad, _, _, _, factor1, factor2 = objective(self._t(batch_x), self._t(batch_y), noise_list)
            train_step.zero_grad()
            cost.sum().backward()
            train_step.step()
            self._project_prior_std(log_prior_std)

            should_log = i % batches_per_epoch == 0 or i == total_iterations - 1
            should_store = save_dict is not None and i % save_dict["iter"] == 0
            if not should_log and not should_store:
                continue

            train_accuracy_stoch = None
            # Added fast mode: eval_interval=0 skips epoch-wide accuracy scans,
            # but does not skip optimization or the separate final evaluation.
            # Missing diagnostics stay None rather than reusing stale accuracy.
            # These diagnostics use the same updated parameters as the KL below.
            # Skipping them also removes their random draws, so training random
            # streams need not match an otherwise identical run with diagnostics.
            if should_log and eval_interval > 0 and epoch % eval_interval == 0:
                train_accuracy_stoch, train_accuracy_det = 0.0, 0.0
                for ib in range(batches_per_epoch):
                    bx, by = next_batch(trainX, trainY, batch_size, ib)
                    diagnostic_noise = generate_noise(self.layer_shapes)
                    train_accuracy_stoch += self._pacb_correct_sum(
                        bx, by, log_post_std_list, diagnostic_noise)
                    train_accuracy_det += self._pacb_correct_sum(
                        bx, by, log_post_std_list, generate_zero_noise(self.layer_shapes))
                train_accuracy_stoch /= Nsamples
                train_accuracy_det /= Nsamples
                mean_accuracy_stoch_print.append(train_accuracy_stoch)
                mean_accuracy_det_print.append(train_accuracy_det)
            # Recompute diagnostics after the optimizer step. Mixing a pre-step
            # KL with post-step means/stds can make saved metrics inconsistent.
            # Reuse the update's noise only for this objective diagnostic; it
            # is not the independent Monte Carlo stream for the final certificate.
            with torch.no_grad():
                A, cost, Bquad, _, _, _, factor1, factor2 = objective(
                    self._t(batch_x), self._t(batch_y), noise_list)
            # Read scalar tensors only when logging or storing a snapshot.
            A_i, cost_i, kldiv2_i, B_i = A.item(), cost.item(), self.KLdivTimes2.item(), self.B.item()
            m_w, v_w = self.mean_weights_component.item(), self.var_weights_component.item()
            _log_prior_std, Bquad_i = log_prior_std.item(), Bquad.item()
            factor1_i, factor2_i = factor1.item(), factor2.item()


            if should_log:
                # History uses a plug-in sampled error and the continuous prior.
                # It omits the final MC correction and discrete-prior selection,
                # so these history values are explicitly diagnostic, not certified.
                bpac = (approximate_BPAC_bound(train_accuracy_stoch, B_i)
                        if train_accuracy_stoch is not None else None)
                self.history.append({
                    "epoch": epoch,
                    "A_term": A_i,
                    "B_term": B_i,
                    "objective": cost_i,
                    "stochastic_train_accuracy": train_accuracy_stoch,
                    "stochastic_train_error": (1.0 - train_accuracy_stoch
                                               if train_accuracy_stoch is not None else None),
                    "training_accuracy_evaluated": train_accuracy_stoch is not None,
                    "KL": kldiv2_i / 2,
                    "PAC_bound_estimate": bpac,
                    "bound_is_certified": False,
                    "log_prior_std": _log_prior_std,
                    "learning_rate": lr_factor * learning_rate if lr_dropped else learning_rate
                })
                accuracy_text = ("%.4f" % train_accuracy_stoch
                                 if train_accuracy_stoch is not None else "not evaluated")
                bound_text = "%.4f" % bpac if bpac is not None else "not evaluated"
                output ="".join("Epoch:" + '%04d' % (epoch+1) + " cost=" + str(cost_i) +
                                " mean accuracy " + accuracy_text + ' KL div:  %.4f' % (kldiv2_i/2) +
                                ' A term: %.4f' % A_i + ' B term: %.4f' % B_i + ' Bquad: %.4f' % Bquad_i +
                                ' log_prior_std: %.4f' % _log_prior_std + ' B PAC: ' + bound_text + ' factor1: %.4f' % factor1_i +
                                ' factor2: %.4f' % factor2_i + f' Iteration: {i + 1}/{total_iterations}')
                print(output)
            if should_store:
                self.PACB_store(save_dict, i=i, log_prior_std=np.array([_log_prior_std], dtype=np.float32), m_w=m_w,
                                v_w=v_w, bpac=bpac, log_post_std_list=log_post_std_list,
                                KL_val=kldiv2_i / 2)

        # Extract all posterior log standard deviations and the prior log std
        # from the final optimizer state. Final evaluation must pair these with
        # the final means, not an earlier periodically stored variance snapshot.
        self.log_prior_std = log_prior_std.detach().cpu().numpy()
        noisevecs = []
        for log_post_std in log_post_std_list:
          noisevecs.append(log_post_std.detach().cpu().numpy())
        self.log_post_all = []
        for nv in noisevecs:
            for v in nv.flatten():
                self.log_post_all.append(v)

        return

    def evaluate_SNN_accuracy(self, testX, testY, prior_weights=None, N_SNN_samples=200, save_dict=None,
                              mc_seed=None):
        """Evaluate one fixed posterior and apply the paper's two KL inversions."""
        if N_SNN_samples < 1 or int(N_SNN_samples) != N_SNN_samples:
            raise ValueError("N_SNN_samples must be a positive integer")
        if prior_weights is None:
            raise ValueError("The data-independent prior mean is required")
        params_mean_values = self.get_model_weights()
        # Reject missing/infinite posterior variances before reconstruction.
        # Evaluating final means with stale or incomplete log stds would evaluate
        # a different Q and can silently invalidate the reported KL and error.
        nparams, self.layer_shapes = self.count_N_params()
        flat_logs = np.asarray(self.log_post_all, dtype=np.float64)
        if flat_logs.ndim != 1 or flat_logs.size != nparams or not np.isfinite(flat_logs).all():
            raise ValueError("A complete finite posterior log-standard-deviation vector is required")
        log_post_std_list = []
        offset = 0
        for shape in self.layer_shapes:
            size = int(np.prod(shape))
            log_post_std_list.append(flat_logs[offset:offset + size].reshape(shape).copy())
            offset += size
        rho = float(np.asarray(self.log_prior_std).reshape(-1)[0])
        # Training treats the prior variance as continuous. The paper's final
        # union-bound certificate requires a discrete positive grid index, so
        # evaluate both valid neighboring priors and retain the tighter bound.
        # Gaussian KL is analytic; posterior MC samples do not estimate this KL.
        selected = discretized_prior_bound(params_mean_values, prior_weights, log_post_std_list, rho,
                                           self.X.shape[0], self.deltaPAC,
                                           self.log_prior_std_precision, self.log_prior_std_base)

        # Use a fresh local RNG independent of the optimized posterior's training
        # perturbations. Do not reset the global training seed and replay draws.
        # The Gaussian sampling law itself is unchanged: w = mu + sigma*epsilon,
        # epsilon ~ N(0, I). Record mc_seed to reproduce this final evaluation;
        # an explicitly supplied seed must also be independent of training.
        if mc_seed is None:
            mc_seed = int(np.random.SeedSequence().entropy)
        rng = np.random.default_rng(mc_seed)
        scale_list = [self._t(np.exp(v)) for v in log_post_std_list]
        mean_train_accuracy, mean_test_accuracy = 0.0, 0.0
        print(f"Final posterior evaluation started: {N_SNN_samples} samples", flush=True)
        for ns in range(N_SNN_samples):
            # One independent posterior draw is shared across all train/test
            # examples, as in the upstream final evaluator. A draw's full-set
            # empirical error is a bounded [0,1] random variable. Thus N draws
            # count as N MC samples, not N times the number of training examples.
            noise_list = [rng.standard_normal(shape).astype(np.float32) for shape in self.layer_shapes]
            mean_train_accuracy += self.print_accuracy_in_batches_noise(
                self.X, self.Y, noise_list, scale_list, whichset='train', verbose=False)
            mean_test_accuracy += self.print_accuracy_in_batches_noise(
                testX, testY, noise_list, scale_list, whichset='test', verbose=False)
            if (ns + 1) % 100 == 0 or ns == N_SNN_samples - 1:
                print(f"Final posterior evaluation: {ns + 1}/{N_SNN_samples}")
        mean_train_accuracy /= N_SNN_samples
        mean_test_accuracy /= N_SNN_samples
        sampled_error = min(1.0, max(0.0, 1 - mean_train_accuracy))
        # Main final-bound correction: upstream directly plugs the sampled
        # error into the PAC inversion. Its SamplesConvBound helper is defined
        # but never called. The paper's Eq. (6) instead first upper-bounds the
        # expected empirical error under Q using independent posterior draws:
        # q_upper = kl_inverse(q_sampled, log(2/deltaMC) / N).
        # Increasing N shrinks this MC penalty, not the chosen confidence delta.
        error_upper = monte_carlo_error_upper(sampled_error, N_SNN_samples, self.deltaMC)
        # The second inversion combines q_upper with analytic Gaussian KL and
        # the data/prior-grid penalties. The final error bound is not simply B
        # or q+B. The two failure probabilities add by the union bound, giving
        # per-experiment confidence 0.965 at the fixed deltas above for any N.
        bpac = inverse_binary_kl(error_upper, selected["relative entropy bound"])
        B_val, KL_val = selected["generalization/complexity term B"], selected["KL(Q || P)"]
        # Retain the uncorrected plug-in result for comparison with the public
        # code. It must not be presented as having the corrected MC confidence.
        # Test error remains a descriptive sampled mean, not a certified interval.
        self.final_evaluation = dict(selected, **{
            "bound implementation": "corrected_pacbayes_v2",
            "SNN train error upper bound": error_upper,
            "PAC-Bayes bound without MC correction": inverse_binary_kl(sampled_error, selected["relative entropy bound"]),
            "MC delta": self.deltaMC,
            "total failure probability": self.deltaPAC + self.deltaMC,
            "confidence": 1 - self.deltaPAC - self.deltaMC,
            "MC seed": int(mc_seed),
            "MC sampling unit": "one posterior draw evaluated on the full training set",
        })
        # Save one complete final posterior: means, posterior log stds, fixed
        # prior mean, and continuous prior log std from this same state.
        # Legacy periodic lists can have different lengths/timestamps; choosing
        # the last entry of each independently can mix incompatible parameters.
        # Reevaluation reads this versioned record and changes only MC estimates,
        # not the trained posterior or the analytically computed Gaussian KL.
        self.output_dict["final_posterior"] = {
            "version": 2,
            "mean_weights": params_mean_values,
            "log_post_std": [v.copy() for v in log_post_std_list],
            "prior_weights": [np.asarray(v).copy() for v in prior_weights],
            "log_prior_std": rho,
            "architecture": list(self.layers),
            "seed": self.seed,
            "model": type(self).__name__.lower(),
            "device": str(self.device),
            "binary": self.Y.shape[-1] == 1,
        }
        self.output_dict["final_evaluation"] = self.final_evaluation.copy()
        print("Sampled train error:", sampled_error, "Sampled test error:", 1 - mean_test_accuracy)
        print("Train error upper bound:", error_upper,
              "Confidence:", self.final_evaluation["confidence"])
        print("PAC bound error:", bpac, "Gen bound:", B_val, "KL value:", KL_val)
        self.PACB_store(save_dict, i=0, log_prior_std=np.asarray(self.log_prior_std).reshape(-1).copy(), bpac=bpac,
                        B_val=B_val, KL_val=KL_val, test_acc=mean_test_accuracy,
                        train_acc=mean_train_accuracy, log_post_std_list=[self._t(v) for v in log_post_std_list])
        return bpac, B_val, KL_val, self.deltaPAC, mean_train_accuracy, mean_test_accuracy, self.log_prior_std
