import os
import glob
import json
import csv
import pandas as pd
import matplotlib.pyplot as plt

def generate_plots_and_tables(results_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate Table 1 comparison
    table1_data = []
    baseline_dir = os.path.join(results_dir, "baseline")
    if os.path.exists(baseline_dir):
        for exp in os.listdir(baseline_dir):
            exp_dir = os.path.join(baseline_dir, exp, "seed11")
            summary_file = os.path.join(exp_dir, "final_summary.json")
            if os.path.exists(summary_file):
                with open(summary_file, "r") as f:
                    summary = json.load(f)
                table1_data.append({
                    "Experiment": exp,
                    "Parameters": summary.get("number of parameters", 0),
                    "SGD Train Error": f"{summary.get('SGD train error', 0.0):.4f}" if summary.get('SGD train error') != "NA" else "NA",
                    "SGD Test Error": f"{summary.get('SGD test error', 0.0):.4f}" if summary.get('SGD test error') != "NA" else "NA",
                    "SNN Test Error": f"{summary.get('SNN test error', 0.0):.4f}",
                    "PAC-Bayes Bound": f"{summary.get('PAC-Bayes bound', 0.0):.4f}",
                    "KL": f"{summary.get('KL(Q || P)', 0.0):.4f}",
                    "Prior Std": f"{summary.get('final log prior std', 0.0):.4f}"
                })
    
    if table1_data:
        df = pd.DataFrame(table1_data)
        # Sort by parameters, though R600 is special
        df.to_csv(os.path.join(output_dir, "table1_comparison.csv"), index=False)
        print("Generated table1_comparison.csv")

    # Plot history for T600 baseline
    t600_history_file = os.path.join(baseline_dir, "T600", "seed11", "history.csv")
    if os.path.exists(t600_history_file):
        df_hist = pd.read_csv(t600_history_file)
        
        # Plot PAC Bound and Stochastic Test Error
        plt.figure(figsize=(10, 6))
        plt.plot(df_hist['epoch'], df_hist['PAC_bound_estimate'], label='PAC Bound', color='blue')
        plt.plot(df_hist['epoch'], df_hist['stochastic_train_error'], label='Stochastic Train Error', color='green')
        plt.title("PAC-Bayes Optimization: T-600")
        plt.xlabel("Epoch")
        plt.ylabel("Error / Bound")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "t600_pac_bound_vs_epoch.png"))
        plt.close()

        # Plot KL Divergence
        plt.figure(figsize=(10, 6))
        plt.plot(df_hist['epoch'], df_hist['KL'], label='KL(Q||P)', color='red')
        plt.title("KL Divergence vs Epoch: T-600")
        plt.xlabel("Epoch")
        plt.ylabel("KL Divergence")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "t600_kl_vs_epoch.png"))
        plt.close()
        print("Generated T-600 baseline plots")

if __name__ == "__main__":
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    generate_plots_and_tables(results_dir, results_dir)
