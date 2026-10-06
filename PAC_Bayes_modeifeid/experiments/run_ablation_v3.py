import os
import sys
import subprocess
import multiprocessing

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SGD_SCRIPT = os.path.join(PKG_ROOT, "snn", "experiments", "run_sgd.py")
PACB_SCRIPT = os.path.join(PKG_ROOT, "snn", "experiments", "run_pacb.py")

def run_experiment(name, seed, out_dir, is_ablation_c=False):
    os.makedirs(out_dir, exist_ok=True)
    
    # Ablation C asks for no SGD pretraining. We achieve this elegantly by running SGD for 0 epochs.
    # This generates the random checkpoint that run_pacb.py expects without altering core parsing files.
    sgd_epochs_to_run = 0 if is_ablation_c else 20
    
    checkpoint_path = os.path.join(out_dir, "checkpoint.pickle")
    
    if not os.path.exists(checkpoint_path):
        if is_ablation_c:
            print(f"[{out_dir}] Generating pure random init checkpoint (0 epochs SGD)...")
        else:
            print(f"[{out_dir}] Running {sgd_epochs_to_run} epochs SGD...")
        sgd_cmd = f'"{sys.executable}" "{SGD_SCRIPT}" fc --layers 600 --sgd_epochs {sgd_epochs_to_run} --binary --seed {seed} --output_dir "{out_dir}" --run_name {name}'
        subprocess.run(sgd_cmd, shell=True, check=True)
    else:
        print(f"[{out_dir}] SGD Checkpoint found, reusing...")
        
    pacb_args = "--pacb_epochs 1000 --lr 0.001 --drop_lr 250 --lr_factor 0.1 --trainw --snn_samples 1"
    
    if not is_ablation_c:
        # For Ablation B (no_trainw)
        pacb_args = pacb_args.replace("--trainw", "--no-trainw")
    
    print(f"[{out_dir}] Running PACB 1000 epochs...")
    pacb_cmd = f'"{sys.executable}" "{PACB_SCRIPT}" fc --layers 600 --sgd_epochs {sgd_epochs_to_run} {pacb_args} --binary --seed {seed} --output_dir "{out_dir}" --run_name {name}'
    subprocess.run(pacb_cmd, shell=True, check=True)

if __name__ == "__main__":
    seed = 11
    base_out = os.path.join(PKG_ROOT, "results")
    print("=== Running Custom Concurrent Ablation Suite v3 ===")
    
    # 1. Ablation B: no-trainw
    ab_b_dir = os.path.join(base_out, "ablation", "no_trainw", f"seed{seed}")
    os.makedirs(ab_b_dir, exist_ok=True)
    
    # 2. Ablation C: pure random init (0 epochs SGD)
    ab_c_dir = os.path.join(base_out, "ablation", "no_sgd", f"seed{seed}")
    os.makedirs(ab_c_dir, exist_ok=True)
    
    print("Starting multiprocessing...")
    
    # is_ablation_c flag passed directly
    p1 = multiprocessing.Process(target=run_experiment, args=("T600", seed, ab_b_dir, False))
    p2 = multiprocessing.Process(target=run_experiment, args=("T600", seed, ab_c_dir, True))
    
    p1.start()
    p2.start()
    
    p1.join()
    p2.join()
    print("=== All concurrent ablations finished! ===")
