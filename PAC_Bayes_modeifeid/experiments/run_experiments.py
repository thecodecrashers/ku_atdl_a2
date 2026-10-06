import os
import sys
import argparse
import subprocess

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SGD_SCRIPT = os.path.join(PKG_ROOT, "snn", "experiments", "run_sgd.py")
PACB_SCRIPT = os.path.join(PKG_ROOT, "snn", "experiments", "run_pacb.py")

ARCHITECTURES = {
    "T600": "--layers 600",
    "T1200": "--layers 1200",
    "T300x2": "--layers 300 300",
    "T600x2": "--layers 600 600",
    "T1200x2": "--layers 1200 1200",
    "T600x3": "--layers 600 600 600",
    "R600": "--layers 600"
}

def run_experiment(name, seed, out_dir, ablation_args=None, is_r600=False):
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Run SGD
    checkpoint_path = os.path.join(out_dir, "checkpoint.pickle")
    if not os.path.exists(checkpoint_path):
        print(f"Running SGD for {name} (seed {seed})...")
        sgd_cmd = f'"{sys.executable}" "{SGD_SCRIPT}" fc {ARCHITECTURES[name]} --sgd_epochs 20 --binary{" --random_labels" if is_r600 else ""} --seed {seed} --output_dir "{out_dir}" --run_name {name}'
        subprocess.run(sgd_cmd, shell=True, check=True)
    else:
        print(f"SGD Checkpoint found for {name} (seed {seed}), reusing...")
        
    # 2. Run PACB
    pacb_args = "--pacb_epochs 1000 --lr 0.001 --drop_lr 250 --lr_factor 0.1 --trainw --snn_samples 1"
    if ablation_args:
        # Override default PACB args with ablation args if present
        for arg, val in ablation_args.items():
            if val is False:
                # E.g., trainw=False -> --no-trainw
                pacb_args = pacb_args.replace("--trainw", "--no-trainw")
            elif arg == "--pacb_epochs":
                pacb_args = pacb_args.replace("--pacb_epochs 1000", f"--pacb_epochs {val}")
            elif arg == "--snn_samples":
                pacb_args = pacb_args.replace("--snn_samples 1", f"--snn_samples {val}")
    
    print(f"Running PACB for {name} (seed {seed})...")
    pacb_cmd = f'"{sys.executable}" "{PACB_SCRIPT}" fc {ARCHITECTURES[name]} --sgd_epochs 20 {pacb_args} --binary{" --random_labels" if is_r600 else ""} --seed {seed} --output_dir "{out_dir}" --run_name {name}'
    subprocess.run(pacb_cmd, shell=True, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["baseline", "ablation", "all"])
    parser.add_argument("--experiment", type=str)
    args = parser.parse_args()
    
    seed = 11
    base_out = os.path.join(PKG_ROOT, "results")
    
    if args.suite in ["baseline", "all"]:
        print("=== Running Baseline Suite ===")
        for exp in ["T600", "T1200", "T300x2", "T600x2", "T1200x2", "T600x3", "R600"]:
            run_experiment(exp, seed, os.path.join(base_out, "baseline", exp, f"seed{seed}"), is_r600=(exp=="R600"))
            
    if args.suite in ["ablation", "all"]:
        print("=== Running Ablation Suite ===")
        # Reusing T600 SGD checkpoint
        sgd_src = os.path.join(base_out, "baseline", "T600", f"seed{seed}", "checkpoint.pickle")
        
        # Ablation A: 5000 epochs
        ab_a_dir = os.path.join(base_out, "ablation", "pacb_epochs_5000", f"seed{seed}")
        os.makedirs(ab_a_dir, exist_ok=True)
        if os.path.exists(sgd_src) and not os.path.exists(os.path.join(ab_a_dir, "checkpoint.pickle")):
            import shutil
            shutil.copy(sgd_src, os.path.join(ab_a_dir, "checkpoint.pickle"))
        run_experiment("T600", seed, ab_a_dir, ablation_args={"--pacb_epochs": 5000})

        # Ablation B: no-trainw
        ab_b_dir = os.path.join(base_out, "ablation", "no_trainw", f"seed{seed}")
        os.makedirs(ab_b_dir, exist_ok=True)
        if os.path.exists(sgd_src) and not os.path.exists(os.path.join(ab_b_dir, "checkpoint.pickle")):
            import shutil
            shutil.copy(sgd_src, os.path.join(ab_b_dir, "checkpoint.pickle"))
        run_experiment("T600", seed, ab_b_dir, ablation_args={"--trainw": False})

        # Ablation C: SNN samples 200
        ab_c_dir = os.path.join(base_out, "ablation", "snn_samples_200", f"seed{seed}")
        os.makedirs(ab_c_dir, exist_ok=True)
        if os.path.exists(sgd_src) and not os.path.exists(os.path.join(ab_c_dir, "checkpoint.pickle")):
            import shutil
            shutil.copy(sgd_src, os.path.join(ab_c_dir, "checkpoint.pickle"))
        run_experiment("T600", seed, ab_c_dir, ablation_args={"--snn_samples": 200})
        
    if args.experiment:
        run_experiment(args.experiment, seed, os.path.join(base_out, "custom", args.experiment, f"seed{seed}"), is_r600=(args.experiment=="R600"))
