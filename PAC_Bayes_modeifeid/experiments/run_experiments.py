import os
import sys
import argparse
import subprocess
import uuid

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SGD_SCRIPT = os.path.join(PKG_ROOT, "snn", "experiments", "run_sgd.py")
PACB_SCRIPT = os.path.join(PKG_ROOT, "snn", "experiments", "run_pacb.py")

ARCHITECTURES = {
    "T600": [600],
    "T1200": [1200],
    "T300x2": [300, 300],
    "T600x2": [600, 600],
    "T1200x2": [1200, 1200],
    "T600x3": [600, 600, 600],
    "R600": [600]
}

def run_experiment(name, seed, out_dir, ablation_args=None, is_r600=False, snn_samples=1, eval_interval=1):
    if os.path.exists(os.path.join(out_dir, "final_summary.json")):
        raise FileExistsError("Existing results are protected; choose a new output directory")
    os.makedirs(out_dir, exist_ok=True)
    ablation_args = ablation_args or {}
    skip_sgd = ablation_args.get("--no_sgd", False)
    sgd_epochs = 0 if skip_sgd else 20
    trainw = ablation_args.get("--trainw", True)
    # Pass an argument list rather than a shell command string. Layer sizes stay
    # separate integer arguments and Windows paths with spaces are preserved.
    model_args = ["fc", "--layers", *map(str, ARCHITECTURES[name]),
                  "--sgd_epochs", str(sgd_epochs), "--binary", "--seed", str(seed),
                  "--output_dir", out_dir, "--run_name", name]
    if is_r600:
        model_args.append("--random_labels")

    # Always regenerate random initialization to avoid reusing legacy bias patches.
    checkpoint_path = os.path.join(out_dir, "checkpoint.pickle")
    if skip_sgd or not os.path.exists(checkpoint_path):
        print(f"Preparing {name} with {sgd_epochs} SGD epochs (seed {seed})...", flush=True)
        sgd_cmd = [sys.executable, SGD_SCRIPT, *model_args]
        if skip_sgd:
            sgd_cmd.append("--overwrite")
        subprocess.run(sgd_cmd, check=True)
    else:
        print(f"SGD checkpoint found for {name} (seed {seed}), reusing...", flush=True)

    # Use the configured reproduction hyperparameters consistently across
    # architectures. These are the local runner's epoch schedule; the paper
    # describes its optimization budget in iterations, so they are not an
    # assertion that every original paper training setting is reproduced.
    pacb_args = ["--pacb_epochs", "1000", "--lr", "0.001", "--drop_lr", "250",
                 "--lr_factor", "0.1", "--trainw" if trainw else "--no-trainw",
                 "--snn_samples", str(snn_samples), "--eval_interval", str(eval_interval)]
    print(f"Running PACB for {name} (seed {seed})...", flush=True)
    pacb_cmd = [sys.executable, PACB_SCRIPT, *model_args, *pacb_args]
    subprocess.run(pacb_cmd, check=True)


def run_ablation_suite(seed=None, base_out=None, snn_samples=1):
    """Run all three concurrent ablations in a new isolated directory."""
    from pathlib import Path
    import uuid

    sys.path.insert(0, PKG_ROOT)
    from experiments.run_ablation_fast import main as run_fast

    argv = ["--snn_samples", str(snn_samples)]
    if seed is not None:
        argv.extend(["--seed", str(seed)])
    if base_out is not None:
        output_dir = Path(base_out) / f"ablation_init_{uuid.uuid4().hex[:8]}"
        argv.extend(["--output_dir", str(output_dir)])
    if run_fast(argv) != 0:
        raise RuntimeError("An initialization ablation failed; see the run logs")


def main(argv=None):
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--suite", choices=["baseline", "ablation", "all"])
    selection.add_argument("--experiment", choices=ARCHITECTURES)
    parser.add_argument("--output_dir", help="New run root; default creates a unique directory")
    parser.add_argument("--snn_samples", type=int, default=1,
                        help="Independent posterior draws for final evaluation; 1 is fast but gives a loose bound")
    parser.add_argument("--eval_interval", type=int, default=1)
    args = parser.parse_args(argv)
    if args.snn_samples < 1 or args.eval_interval < 0:
        parser.error("snn_samples must be positive and eval_interval must be nonnegative")
    
    # Preserve the baseline's training seed. The separate fast ablation runner
    # chooses a fresh seed unless explicitly instructed otherwise.
    seed = 11
    base_out = (args.output_dir or os.environ.get("PACB_RUN_ROOT") or
                os.path.join(PKG_ROOT, "results", f"run_{uuid.uuid4().hex}"))
    base_out = os.path.abspath(base_out)
    try:
        # Allocate a new run root once, before any training subprocess starts.
        # Existing experiment folders remain historical evidence, never targets
        # for an implicit restart or a revised bound calculation.
        os.makedirs(base_out, exist_ok=False)
    except FileExistsError:
        parser.error(f"Existing run roots are protected; choose a new output_dir: {base_out}")
    print(f"New run root: {base_out}", flush=True)
    
    if args.suite in ["baseline", "all"]:
        print("=== Running Baseline Suite ===")
        for exp in ["T600", "T1200", "T300x2", "T600x2", "T1200x2", "T600x3", "R600"]:
            run_experiment(exp, seed, os.path.join(base_out, "baseline", exp, f"seed{seed}"), is_r600=(exp=="R600"), snn_samples=args.snn_samples, eval_interval=args.eval_interval)
            
    if args.suite in ["ablation", "all"]:
        run_ablation_suite(base_out=base_out, snn_samples=args.snn_samples)
        
    if args.experiment:
        run_experiment(args.experiment, seed, os.path.join(base_out, "custom", args.experiment, f"seed{seed}"), is_r600=(args.experiment=="R600"), snn_samples=args.snn_samples, eval_interval=args.eval_interval)


if __name__ == "__main__":
    main()
