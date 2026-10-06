import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from snn.core import package_path
from snn.core.parse_args import BasicParser, Interpreter
from snn.core.utils import serialize


def run_sgd(model, epochs, testX, testY, output_dir=None, run_name=None):
    """
        Runs SGD for a predefined number of epochs and saves the resulting model.
    """
    print("Training full network")
    weights_rand_init = model.optimize(epochs=epochs)
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
                "SGD epochs": epochs,
                "SGD learning rate": 0.01,
                "momentum": 0.9,
                "batch size": 100
            }, f, indent=4)

    return [model.get_model_weights(), weights_rand_init]


if __name__ == '__main__':
    basic_args = BasicParser().parse()
    model, test_set, save_path = Interpreter(basic_args).interpret()
    saved_entities = run_sgd(model, basic_args["sgd_epochs"], test_set[0], test_set[1], basic_args.get("output_dir"), basic_args.get("run_name"))
    
    # Still save to original path for PACB reuse, unless an output_dir is handling everything
    serialization_path = os.path.join(package_path, "experiments", save_path)
    if basic_args.get("output_dir"):
        serialization_path = os.path.join(basic_args["output_dir"], "checkpoint.pickle")
    
    print("Saving run in ", serialization_path)
    serialize(saved_entities, serialization_path, basic_args["overwrite"])
    print("SGD run complete!")
