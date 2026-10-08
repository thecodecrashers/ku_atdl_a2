# PAC-Bayes bound corrections

The baseline previously reported training KL of about 384 and final KL of about 97,643. The training objective cached the distance between the SGD initialization and the prior, while final evaluation used the optimized mean. Final inverse KL arithmetic was consistent with its inputs; the training objective was missing the current-mean penalty and its gradient.

## Objective and prior

The objective now evaluates the diagonal Gaussian KL using the current posterior mean on every mini-batch. With `rho = log(prior_std)` and `r_i = 2 * (log(posterior_std_i) - rho)`, it computes:

```text
2 KL(Q || P) = ||w - w0||^2 exp(-2 rho) + sum(expm1(r_i) - r_i)
```

This is algebraically equivalent to the Gaussian KL in the [paper, equations (1) and (5)](https://arxiv.org/html/1703.11008). The stable covariance expression reduces cancellation. Mean parameters remain in the autograd graph for the first two ablations and remain frozen for `no_trainw`.

After each optimizer step, the prior log standard deviation is projected onto `rho <= (log(0.1) - 1/100)/2`. This ensures `j = 100 * (log(0.1) - 2 rho) >= 1`. Final evaluation selects the better of the adjacent positive integer grid indices, using `lambda_j = 0.1 exp(-j/100)` and its actual union-bound penalty. Invalid legacy prior values are rejected instead of using an absolute index. Training and final evaluation both use PAC-Bayes delta `0.025`.

The training log uses post-update KL, posterior variance, and prior variance from the same state. History bounds are diagnostic estimates, explicitly marked `bound_is_certified=False`; they omit the final Monte Carlo confidence correction and prior discretization.

## Final evaluation

For `N` independent posterior draws, each draw is evaluated on the complete training set. Its empirical classification error is a bounded random variable in `[0, 1]`. The KL Chernoff bound for bounded independent variables gives the conservative correction:

```text
train_error_upper = inverse_KL(sampled_train_error, log(2 / 0.01) / N)
PAC_bound = inverse_KL(train_error_upper,
    (KL(Q || P_j) + log(pi^2 m / (6 * 0.025)) + 2 log(j)) / (m - 1))
```

The failure probability for each final bound is at most `0.025 + 0.01 = 0.035`. The final sampling stream uses fresh entropy independent of training; its seed is recorded for replay. One full-set posterior draw counts as one Monte Carlo sample, not `m` independent samples. The sampled test error is descriptive and does not receive an additional confidence guarantee here.

The fast default remains one draw. Even with zero sampled error, its training error upper bound is `0.995`; it cannot provide a tight final certificate. Use `--snn_samples 1000` for a more useful estimate, or 150,000 draws for the paper's stated sampling count, accounting for evaluation cost.

Inverse KL uses bracketed bisection with explicit zero/one endpoints and returns the upper bracket. Binary logistic loss uses `softplus` to avoid overflow. Accuracy aggregation weights batches by example count.

## Results and checkpoints

Existing result files are preserved. Fresh baseline and ablation runs use new directories; existing final summaries are protected. Each new model file has an authoritative `final_posterior` record containing the matching mean, posterior log standard deviations, prior mean, continuous prior log standard deviation, and model metadata. `final_evaluation` records the selected discrete prior, Monte Carlo seed, and confidence parameters.

`recompute_summary.py` reads this record and writes `final_summary_recomputed.json`. It refuses incomplete legacy snapshots and existing target files. A new second-stage training run is required to repair results produced by the old objective. Final continuous training KL and discretized evaluation KL can differ slightly because their prior variances differ.

The original TensorFlow folder and other experiment hyperparameters are preserved. This correction does not claim a completed full Table 1 reproduction.

## Validation

```powershell
python PAC_Bayes_modeifeid/tests/test_pacbayes_correctness.py -v
```

The regression checks compare Gaussian KL against `torch.distributions`, verify its mean gradient analytically, test positive prior indices and inverse KL endpoints, check frozen weights and skipped epoch evaluation, and reproduce final summaries from complete saved snapshots while preserving the original summary.

All 12 regression checks passed. A real three-process CPU smoke run used T-600, one PAC-Bayes epoch, three final posterior draws, and one SGD epoch for the frozen condition. Its SGD update counts were 0, 1, and 550; only the frozen condition kept its mean exactly unchanged. Recomputed continuous KL matched the last training log within float32 precision, and discrete final KL matched each saved summary. The full 1,000-epoch experiments were not rerun as part of this correction.
