import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from snn.core import package_path
from snn.core.parse_args import BasicParser, Interpreter
from snn.core.utils import serialize


def run_sgd(model, epochs, testX, testY, output_dir=None, run_name=None, sgd_steps=None):
    """
        Runs SGD for a predefined number of epochs and saves the resulting model.
    """
    print("Training full network")
    # Save both roles explicitly: the post-SGD means initialize stage two,
    # while weights_rand_init records the fixed prior mean from before SGD.
    # sgd_steps caps mini-batch updates for the one-update initialization ablation.
    weights_rand_init = model.optimize(epochs=epochs, max_steps=sgd_steps)
    print("Model optimized!!!")

    train_acc = model.print_accuracy_in_batches(model.X, model.Y, no_batches=50)
    test_acc = model.print_accuracy_in_batches(testX, testY, no_batches=50, whichset='test')
    
    train_err = 1.0 - train_acc
    test_err = 1.0 - test_acc
    n_params = model.count_N_params()[0]
    
    print("\n================ SGD RESULT ================")
    print(f"Experiment: {run_name}")
    print(f"Architecture: {model.layers}")
    print(f"Seed: {model.seed}")
    print(f"Parameters: {n_params}")
    print(f"Train error: {train_err:.4f}")
    print(f"Test error: {test_err:.4f}")
    print("============================================\n")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "sgd_metrics.json"), "w") as f:
            json.dump({
                "SGD train accuracy": train_acc,
                "SGD train error": train_err,
                "SGD test accuracy": test_acc,
                "SGD test error": test_err,
                "number of parameters": n_params,
                "architecture": model.layers,
                "seed": model.seed,
                # A capped warmup may stop partway through an epoch. Report
                # its actual fractional epoch count and update count alongside
                # the requested budget so "one step" is not reported as "one epoch".
                "SGD epochs": (epochs if sgd_steps is None else
                               model.sgd_steps_completed / int(model.Nsamples / 100)),
                "SGD requested epochs": epochs,
                "SGD steps": model.sgd_steps_completed,
                "SGD learning rate": 0.01,
                "momentum": 0.9,
                "batch size": 100
            }, f, indent=4)

    return [model.get_model_weights(), weights_rand_init]


if __name__ == '__main__':
    basic_args = BasicParser().parse()
    model, test_set, save_path = Interpreter(basic_args).interpret()
    saved_entities = run_sgd(model, basic_args["sgd_epochs"], test_set[0], test_set[1], basic_args.get("output_dir"), basic_args.get("run_name"), sgd_steps=basic_args["sgd_steps"])
    
    # Preserve the legacy path for standalone use, but prefer the per-run
    # checkpoint when output_dir is supplied. Concurrent ablations must not
    # share a checkpoint or accidentally load a previously trained baseline.
    serialization_path = os.path.join(package_path, "experiments", save_path)
    if basic_args.get("output_dir"):
        serialization_path = os.path.join(basic_args["output_dir"], "checkpoint.pickle")
    
    print("Saving run in ", serialization_path)
    serialize(saved_entities, serialization_path, basic_args["overwrite"])
    print("SGD run complete!")
