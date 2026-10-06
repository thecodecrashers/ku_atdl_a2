import os
import sys
import subprocess
import json
import csv
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = sys.executable

PAPER_VALUES = {
    "T600": {"SGD train": .001, "SGD test": .018, "SNN train": .028, "SNN test": .034, "PAC bound": .161, "KL": 5144},
    "T1200": {"SGD train": .002, "SGD test": .018, "SNN train": .027, "SNN test": .035, "PAC bound": .179, "KL": 5977},
    "T300x2": {"SGD train": .000, "SGD test": .015, "SNN train": .027, "SNN test": .034, "PAC bound": .170, "KL": 5791},
    "T600x2": {"SGD train": .000, "SGD test": .016, "SNN train": .028, "SNN test": .033, "PAC bound": .186, "KL": 6534},
    "T1200x2": {"SGD train": .000, "SGD test": .015, "SNN train": .029, "SNN test": .035, "PAC bound": .223, "KL": 8558},
    "T600x3": {"SGD train": .000, "SGD test": .013, "SNN train": .027, "SNN test": .032, "PAC bound": .201, "KL": 7861},
    "R600": {"SGD train": .007, "SGD test": .508, "SNN train": .112, "SNN test": .503, "PAC bound": 1.352, "KL": 201131},
}

EXPERIMENT_CONFIGS = {
    "T600": {"layers": [600], "binary": True},
    "T1200": {"layers": [1200], "binary": True},
    "T300x2": {"layers": [300, 300], "binary": True},
    "T600x2": {"layers": [600, 600], "binary": True},
    "T1200x2": {"layers": [1200, 1200], "binary": True},
    "T600x3": {"layers": [600, 600, 600], "binary": True},
    "R600": {"layers": [600], "binary": True, "random_labels": True, "sgd_epochs": 100}, # More epochs for random labels
}

def run_experiment(exp_name, seed, out_dir, sgd_epochs=20, pacb_epochs=1000, lr=0.001, drop_lr=250, lr_factor=0.1, trainw=True, snn_samples=1):
    os.makedirs(out_dir, exist_ok=True)
    cfg = EXPERIMENT_CONFIGS[exp_name]
    layers = " ".join(map(str, cfg["layers"]))
    binary_flag = "--binary" if cfg.get("binary") else ""
    random_flag = "--random_labels" if cfg.get("random_labels") else ""
    actual_sgd_epochs = cfg.get("sgd_epochs", sgd_epochs)
    
    sgd_cmd = f'{PYTHON} "{os.path.join(ROOT, "snn", "experiments", "run_sgd.py")}" fc --layers {layers} --sgd_epochs {actual_sgd_epochs} {binary_flag} {random_flag} --seed {seed} --output_dir "{out_dir}" --run_name {exp_name}'
    
    # Check if SGD needs to run
    checkpoint = os.path.join(out_dir, "checkpoint.pickle")
    if not os.path.exists(checkpoint):
        print(f"Running SGD for {exp_name} (seed {seed})...")
        print(sgd_cmd)
        subprocess.run(sgd_cmd, shell=True, check=True)
    else:
        print(f"Checkpoint exists for {exp_name} (seed {seed}), skipping SGD.")

    trainw_flag = "--trainw" if trainw else "--no-trainw"
    
    pacb_cmd = f'{PYTHON} "{os.path.join(ROOT, "snn", "experiments", "run_pacb.py")}" fc --layers {layers} --sgd_epochs {actual_sgd_epochs} --pacb_epochs {pacb_epochs} --lr {lr} --drop_lr {drop_lr} --lr_factor {lr_factor} {trainw_flag} {binary_flag} {random_flag} --seed {seed} --output_dir "{out_dir}" --run_name {exp_name} --snn_samples {snn_samples}'
    
    print(f"Running PACB for {exp_name} (seed {seed})...")
    print(pacb_cmd)
    subprocess.run(pacb_cmd, shell=True, check=True)

def generate_table1(baseline_dir):
    results = []
    for exp_name in PAPER_VALUES.keys():
        out_dir = os.path.join(baseline_dir, exp_name, "seed11")
        summary_file = os.path.join(out_dir, "final_summary.json")
        sgd_file = os.path.join(out_dir, "sgd_metrics.json")
        if os.path.exists(summary_file) and os.path.exists(sgd_file):
            with open(summary_file) as f:
                s = json.load(f)
            with open(sgd_file) as f:
                sgd = json.load(f)
            
            p = PAPER_VALUES[exp_name]
            results.append({
                "Experiment": exp_name,
                "Paper SGD train error": p["SGD train"],
                "Our SGD train error": round(sgd["SGD train error"], 3),
                "Paper SGD test error": p["SGD test"],
                "Our SGD test error": round(sgd["SGD test error"], 3),
                "Paper SNN train error": p["SNN train"],
                "Our SNN train error": round(s["SNN train error"], 3),
                "Paper SNN test error": p["SNN test"],
                "Our SNN test error": round(s["SNN test error"], 3),
                "Paper KL": p["KL"],
                "Our KL": round(s["KL(Q || P)"]),
                "Paper PAC bound": p["PAC bound"],
                "Our PAC bound": round(s["PAC-Bayes bound"], 3),
                "absolute PAC-bound difference": round(abs(s["PAC-Bayes bound"] - p["PAC bound"]), 3)
            })
    if not results:
        return
        
    csv_path = os.path.join(baseline_dir, "table1_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\\nTable 1 Comparison saved to {csv_path}")

def run_suite_baseline():
    baseline_dir = os.path.join(ROOT, "results", "baseline")
    for exp_name in PAPER_VALUES.keys():
        out_dir = os.path.join(baseline_dir, exp_name, "seed11")
        try:
            run_experiment(exp_name, 11, out_dir)
        except Exception as e:
            print(f"Experiment {exp_name} failed: {e}")
    generate_table1(baseline_dir)

def run_suite_ablation():
    ablation_dir = os.path.join(ROOT, "results", "ablation")
    # Base T600 checkpoint dir for ablation to share SGD
    base_t600_dir = os.path.join(ROOT, "results", "baseline", "T600", "seed11")
    if not os.path.exists(os.path.join(base_t600_dir, "checkpoint.pickle")):
        print("Creating base T600 checkpoint for ablations...")
        run_experiment("T600", 11, base_t600_dir)

    # Ablation A: PACB epochs
    for epochs in [100, 250, 500, 1000]:
        out_dir = os.path.join(ablation_dir, "pacb_epochs", f"{epochs}")
        os.makedirs(out_dir, exist_ok=True)
        # Copy checkpoint
        import shutil
        shutil.copy(os.path.join(base_t600_dir, "checkpoint.pickle"), os.path.join(out_dir, "checkpoint.pickle"))
        shutil.copy(os.path.join(base_t600_dir, "sgd_metrics.json"), os.path.join(out_dir, "sgd_metrics.json"))
        run_experiment("T600", 11, out_dir, pacb_epochs=epochs)

    # Ablation B: trainw
    for trainw in [True, False]:
        out_dir = os.path.join(ablation_dir, "train_posterior_mean", str(trainw))
        os.makedirs(out_dir, exist_ok=True)
        import shutil
        shutil.copy(os.path.join(base_t600_dir, "checkpoint.pickle"), os.path.join(out_dir, "checkpoint.pickle"))
        shutil.copy(os.path.join(base_t600_dir, "sgd_metrics.json"), os.path.join(out_dir, "sgd_metrics.json"))
        run_experiment("T600", 11, out_dir, trainw=trainw)
        
    # Ablation C: SNN samples
    for samples in [1, 20, 100, 200]:
        out_dir = os.path.join(ablation_dir, "snn_samples", str(samples))
        os.makedirs(out_dir, exist_ok=True)
        import shutil
        shutil.copy(os.path.join(base_t600_dir, "checkpoint.pickle"), os.path.join(out_dir, "checkpoint.pickle"))
        shutil.copy(os.path.join(base_t600_dir, "sgd_metrics.json"), os.path.join(out_dir, "sgd_metrics.json"))
        run_experiment("T600", 11, out_dir, snn_samples=samples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["baseline", "ablation", "all"])
    parser.add_argument("--experiment", type=str)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11])
    args = parser.parse_args()

    if args.suite == "baseline" or args.suite == "all":
        run_suite_baseline()
    if args.suite == "ablation" or args.suite == "all":
        run_suite_ablation()
        
    if args.experiment:
        for seed in args.seeds:
            out_dir = os.path.join(ROOT, "results", "custom", args.experiment, f"seed{seed}")
            run_experiment(args.experiment, seed, out_dir)
