import os
import sys
import json
import csv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from snn.core import package_path
from snn.core.parse_args import CompleteParser, Interpreter
from snn.core.utils import deserialize


def run_pacb(weights_rand_init, model, test_set, epochs, learning_rate, drop_lr, lr_factor, seed, trainw, snn_samples, output_dir=None, run_name=None, eval_interval=1, random_labels=False):
    testX, testY = test_set
    if snn_samples < 1:
        raise ValueError("snn_samples must be positive")
    filename = f"model_mean_opt{trainw}_LR{learning_rate}_seed{seed}.pickle"
    path = os.path.join(output_dir or os.path.join(package_path, "experiments", "cifar"), filename)
    # Check for existing outputs before training starts. New experiments must
    # not overwrite old model files, histories, or final summaries, including
    # runs that still use the legacy default output location.
    if os.path.exists(path):
        raise FileExistsError("Existing model output is protected; choose a new output_dir")
    if output_dir:
        protected = ["history.csv", "final_summary.json"]
        if any(os.path.exists(os.path.join(output_dir, name)) for name in protected):
            raise FileExistsError("Existing PAC-Bayes results are protected; choose a new output_dir")

    save_dict = {"log_post_all": True, "PACB_weights": True, "L2_PACB": False, "diff": False, "iter": 500*50,
                 "w*": model.get_model_weights(), "mean_weights": True, "var_weights": True, "PACBound": True,
                 "B_val": True, "KL_val": True, "test_acc": True, "train_acc": True, "log_prior_std": True}
    model.PACB_init
    # The checkpoint contains the stage-one posterior initializer and the
    # original random prior mean separately. trainw controls only posterior
    # mean updates; eval_interval controls optional training diagnostics.
    model.optimize_PACB(weights_rand_init, epochs, learning_rate=learning_rate, drop_lr=drop_lr, lr_factor=lr_factor,
                        save_dict=save_dict, trainWeights=trainw, eval_interval=eval_interval)
    
    # Unlike upstream's hardcoded N=1, evaluate the requested number of fresh
    # posterior draws. This evaluator performs both MC and PAC KL inversions
    # and records a complete final posterior for later reevaluation.
    bpac, B_val, KL_val, deltaPAC, mean_train_accuracy, mean_test_accuracy, log_prior_std = model.evaluate_SNN_accuracy(testX, testY, weights_rand_init, N_SNN_samples=snn_samples, save_dict=save_dict)
    model.output_dict["final_posterior"]["random_labels"] = bool(random_labels)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        # Dump history
        if hasattr(model, "history") and model.history:
            keys = model.history[0].keys()
            with open(os.path.join(output_dir, "history.csv"), "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(model.history)
        
        # Dump final summary
        summary = {
            "Experiment": run_name,
            "architecture": list(model.layers),
            "seed": seed,
            "random labels": bool(random_labels),
            "SNN train accuracy": float(mean_train_accuracy),
            "SNN train error": float(1.0 - mean_train_accuracy),
            "SNN test accuracy": float(mean_test_accuracy),
            "SNN test error": float(1.0 - mean_test_accuracy),
            "PAC-Bayes bound": float(bpac),
            "generalization/complexity term B": float(B_val),
            "KL(Q || P)": float(KL_val),
            "final log prior std": float(log_prior_std.flatten()[0]) if isinstance(log_prior_std, np.ndarray) else float(log_prior_std),
            "number of SNN evaluation samples": snn_samples,
            "training evaluation interval": eval_interval,
            "delta": float(deltaPAC),
            "number of parameters": model.count_N_params()[0]
        }
        # final_evaluation supplies the selected discrete prior, corrected
        # training error upper bound, MC seed, and confidence parameters.
        # Keep these with the sampled means so an estimate cannot be mistaken
        # for a certificate or replayed with a different prior or posterior.
        summary.update(model.final_evaluation)
        with open(os.path.join(output_dir, "final_summary.json"), "x", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)
            
        print("\n============= FINAL PAC-BAYES RESULT =============")
        print(f"Experiment: {run_name}")
        print(f"Architecture: {model.layers}")
        print(f"Seed: {seed}")
        print(f"SGD train error: NA")
        print(f"SGD test error: NA")
        print(f"SNN train error: {1.0 - mean_train_accuracy:.4f}")
        print(f"SNN test error: {1.0 - mean_test_accuracy:.4f}")
        print(f"KL(Q || P): {KL_val:.4f}")
        print(f"Complexity term B: {B_val:.4f}")
        print(f"PAC-Bayes bound: {bpac:.4f}")
        print(f"Non-vacuous: {'YES' if bpac < 1.0 else 'NO'}")
        print("==================================================\n")

    model.save_output(path=path)


if __name__ == '__main__':
    complete_args = CompleteParser().parse()
    _, _, save_path = Interpreter(complete_args).interpret()
    deserialization_path = os.path.join(package_path, "experiments", save_path)
    if complete_args.get("output_dir"):
        potential_path = os.path.join(complete_args["output_dir"], "checkpoint.pickle")
        if os.path.exists(potential_path):
            deserialization_path = potential_path
            
    print("Loading model weights saved in ", deserialization_path)
    model_weights, weights_rand_init = deserialize(deserialization_path)
    print("Model weights loaded!")
    model, test_set, _ = Interpreter(complete_args).interpret(model_weights)
    
    snn_samples = complete_args.get("snn_samples", 1)
    
    run_pacb(weights_rand_init, model, test_set, complete_args["pacb_epochs"], complete_args["lr"],
             complete_args["drop_lr"], complete_args["lr_factor"], complete_args["seed"], complete_args["trainw"], snn_samples, complete_args.get("output_dir"), complete_args.get("run_name"), eval_interval=complete_args["eval_interval"], random_labels=complete_args["random_labels"])
    print("PAC-Bayes run complete!")
