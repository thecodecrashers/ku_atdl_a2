"""Reevaluate a complete final posterior without overwriting existing results."""

import argparse
import json
from pathlib import Path
import pickle
import sys

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))

from snn.core.parse_args import Interpreter


def recompute_summary_for_dir(output_dir, snn_samples=1, summary_path=None, mc_seed=None, device=None):
    # Added evaluation-only workflow: increasing N must not retrain the network
    # or overwrite its original summary. The new certificate uses the same Q,
    # the same prior grid calculation, and new independent MC draws.
    output_dir = Path(output_dir).resolve()
    target = (Path(summary_path).resolve() if summary_path else
              output_dir / "final_summary_recomputed.json")
    if target.exists():
        raise FileExistsError(f"Existing results are protected; choose a new --summary_path: {target}")
    model_pickles = list(output_dir.glob("model_mean_*.pickle"))
    if len(model_pickles) != 1:
        raise ValueError("Expected exactly one model_mean_*.pickle in the output directory")
    with model_pickles[0].open("rb") as stream:
        saved = pickle.load(stream)
    # Read the versioned complete record rather than pairing the final entries
    # of legacy history lists. Those entries can come from different iterations,
    # especially when posterior log stds were not saved at the final update.
    snapshot = saved.get("final_posterior")
    if not snapshot or snapshot.get("version") != 2:
        raise ValueError("Legacy output has no complete final posterior. Do not combine the last entries "
                         "of historical lists; rerun the corrected PAC-Bayes stage in a new directory.")
    args = {
        "model": snapshot["model"], "layers": snapshot["architecture"][1:-1],
        "sgd_epochs": 0, "seed": snapshot["seed"], "binary": snapshot["binary"],
        "device": device or snapshot.get("device"), "random_labels": snapshot.get("random_labels", False),
    }
    model, (testX, testY), _ = Interpreter(args).interpret(snapshot["mean_weights"])
    model.log_post_all = [float(v) for log_std in snapshot["log_post_std"] for v in log_std.flat]
    model.log_prior_std = snapshot["log_prior_std"]
    # Restore the final means, all posterior log stds, and the fixed prior mean
    # before evaluating. Changing N changes the sampled error/MC margin, while
    # analytic Gaussian KL is unchanged for this fixed model and prior choice.
    result = model.evaluate_SNN_accuracy(testX, testY, snapshot["prior_weights"],
                                         N_SNN_samples=snn_samples, mc_seed=mc_seed)
    bpac, B_val, KL_val, delta, train_acc, test_acc, rho = result
    summary = {
        "SNN train accuracy": train_acc, "SNN train error": 1 - train_acc,
        "SNN test accuracy": test_acc, "SNN test error": 1 - test_acc,
        "PAC-Bayes bound": bpac, "generalization/complexity term B": B_val,
        "KL(Q || P)": KL_val, "final log prior std": float(rho),
        "number of SNN evaluation samples": snn_samples, "delta": delta,
        "number of parameters": model.count_N_params()[0],
        "architecture": snapshot["architecture"], "seed": snapshot["seed"],
        "random labels": args["random_labels"],
    }
    # Persist the MC seed and the actual confidence correction so replay and
    # comparison do not silently substitute an uncorrected plug-in bound.
    summary.update(model.final_evaluation)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=4)
    print(f"Recomputed summary: {target}")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True, help="Directory from a corrected PAC-Bayes run")
    parser.add_argument("--snn_samples", type=int, default=1)
    parser.add_argument("--summary_path", help="New summary file; existing files are never overwritten")
    parser.add_argument("--mc_seed", type=int, help="Independent evaluation seed; default uses fresh entropy")
    parser.add_argument("--device", help="Evaluation device; default reuses the saved device")
    args = parser.parse_args(argv)
    try:
        recompute_summary_for_dir(args.output_dir, args.snn_samples, args.summary_path, args.mc_seed, args.device)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
