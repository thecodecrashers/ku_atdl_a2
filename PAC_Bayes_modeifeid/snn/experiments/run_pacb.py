import csv
import json
import os
import sys
import pickle

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from snn.core import package_path
from snn.core.parse_args import CompleteParser, Interpreter
from snn.core.utils import deserialize, serialize


def run_pacb(weights_rand_init, model, test_set, epochs, learning_rate, drop_lr, lr_factor, seed, trainw, snn_samples, output_dir=None, run_name=None):
    testX, testY = test_set

    save_dict = {"log_post_all": True, "PACB_weights": True, "L2_PACB": False, "diff": False, "iter": 500*50,
                 "w*": model.get_model_weights(), "mean_weights": True, "var_weights": True, "PACBound": True,
                 "B_val": True, "KL_val": True, "test_acc": True, "train_acc": True, "log_prior_std": True}
    model.PACB_init
    model.optimize_PACB(weights_rand_init, epochs, learning_rate=learning_rate, drop_lr=drop_lr, lr_factor=lr_factor,
                        save_dict=save_dict, trainWeights=trainw)
    
    bpac, B_val, KL_val, deltaPAC, mean_train_accuracy, mean_test_accuracy, log_prior_std = model.evaluate_SNN_accuracy(testX, testY, weights_rand_init, N_SNN_samples=snn_samples, save_dict=save_dict)

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
            "SNN train accuracy": float(mean_train_accuracy),
            "SNN train error": float(1.0 - mean_train_accuracy),
            "SNN test accuracy": float(mean_test_accuracy),
            "SNN test error": float(1.0 - mean_test_accuracy),
            "PAC-Bayes bound": float(bpac),
            "generalization/complexity term B": float(B_val),
            "KL(Q || P)": float(KL_val),
            "final log prior std": float(log_prior_std.flatten()[0]) if isinstance(log_prior_std, np.ndarray) else float(log_prior_std),
            "number of SNN evaluation samples": snn_samples,
            "delta": float(deltaPAC),
            "number of parameters": model.count_N_params()[0]
        }
        with open(os.path.join(output_dir, "final_summary.json"), "w") as f:
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

        path = os.path.join(output_dir, f"model_mean_opt{trainw}_LR{learning_rate}_seed{seed}.pickle")
        original_model_path = os.path.join(output_dir, f"original_model_mean_opt{trainw}_LR{learning_rate}_seed{seed}.pickle")
    else:
        path = os.path.join(package_path, "experiments", "cifar",
                            ("model_mean_opt{}_LR{}_seed{}.pickle".format(trainw, learning_rate, seed)))
    model.save_output(path=path)
    with open(path, 'wb') as f:
        pickle.dump(model, f)


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
             complete_args["drop_lr"], complete_args["lr_factor"], complete_args["seed"], complete_args["trainw"], snn_samples, complete_args.get("output_dir"), complete_args.get("run_name"))
    print("PAC-Bayes run complete!")
