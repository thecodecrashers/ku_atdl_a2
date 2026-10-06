import os
import sys
import subprocess
import multiprocessing

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SGD_SCRIPT = os.path.join(PKG_ROOT, "snn", "experiments", "run_sgd.py")
PACB_SCRIPT = os.path.join(PKG_ROOT, "snn", "experiments", "run_pacb.py")

def run_experiment(name, seed, out_dir, ablation_args=None):
    os.makedirs(out_dir, exist_ok=True)
    
    skip_sgd = ablation_args and "--no_sgd" in ablation_args
    checkpoint_path = os.path.join(out_dir, "checkpoint.pickle")
    sgd_epochs_to_run = 0 if skip_sgd else 20
    
    if not os.path.exists(checkpoint_path):
        if skip_sgd:
            print(f"[{out_dir}] Generating pure random init checkpoint (0 epochs SGD)...")
        else:
            print(f"[{out_dir}] Running 20 epochs SGD...")
        sgd_cmd = [sys.executable, SGD_SCRIPT, "fc", "--layers", "600", "--sgd_epochs", str(sgd_epochs_to_run), "--binary", "--seed", str(seed), "--output_dir", out_dir, "--run_name", name]
        subprocess.run(sgd_cmd, check=True)
    else:
        print(f"[{out_dir}] SGD Checkpoint found, reusing...")
        
    pacb_args = ["--pacb_epochs", "1000", "--lr", "0.001", "--drop_lr", "250", "--lr_factor", "0.1", "--trainw", "--snn_samples", "1"]
        
    if ablation_args:
        for arg, val in ablation_args.items():
            if val is False and arg == "--trainw":
                pacb_args[pacb_args.index("--trainw")] = "--no-trainw"
    
    print(f"[{out_dir}] Running PACB 1000 epochs...")
    pacb_cmd = [sys.executable, PACB_SCRIPT, "fc", "--layers", "600", "--sgd_epochs", str(sgd_epochs_to_run)] + pacb_args + ["--binary", "--seed", str(seed), "--output_dir", out_dir, "--run_name", name]
    subprocess.run(pacb_cmd, check=True)

if __name__ == "__main__":
    seed = 11
    base_out = os.path.join(PKG_ROOT, "results")
    print("=== Running Custom Concurrent Ablation Suite ===")
    
    # 1. Ablation B: no-trainw
    ab_b_dir = os.path.join(base_out, "ablation", "no_trainw", f"seed{seed}")
    os.makedirs(ab_b_dir, exist_ok=True)
    
    # 2. Ablation C: no-sgd
    ab_c_dir = os.path.join(base_out, "ablation", "no_sgd", f"seed{seed}")
    os.makedirs(ab_c_dir, exist_ok=True)
    
    print("Starting multiprocessing...")
    
    p1 = multiprocessing.Process(target=run_experiment, args=("T600", seed, ab_b_dir, {"--trainw": False}))
    p2 = multiprocessing.Process(target=run_experiment, args=("T600", seed, ab_c_dir, {"--no_sgd": True}))
    
    p1.start()
    p2.start()
    
    p1.join()
    p2.join()
    print("=== All concurrent ablations finished! ===")
