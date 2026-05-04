from pathlib import Path
import re

import pandas as pd
import matplotlib.pyplot as plt


def plot_drug_reward_boxplots(results_folder):
    results_folder = Path(results_folder)

    def extract_reward(path):
        #Extract the reward value from the filename
        match = re.search(r"drug_reward_(\d+)", path.stem)
        return int(match.group(1)) if match else float("inf")

    # Get and sort CSV files
    csv_files = sorted(
        results_folder.glob("*.csv"),
        key=extract_reward
    )

    # Labels like: Reward 5, Reward 25, Reward 100
    rewards = [extract_reward(f) for f in csv_files]
    labels = [f"Reward {r}" for r in rewards]

    # Load data
    dfs = [pd.read_csv(f) for f in csv_files]

    metrics = [
        "Drugs_Consumed",
        "Food_Consumed",
        "Total_Reward",
    ]

    # Create one figure per metric
    for metric in metrics:
        data = [df[metric].dropna() for df in dfs]

        plt.figure(figsize=(7, 5))
        plt.boxplot(data, labels=labels, showmeans=True)

        plt.title(f"{metric} by Drug Reward")
        plt.xlabel("Drug Reward")
        plt.ylabel(metric)
        plt.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    plot_drug_reward_boxplots("../Results/Drugs_No_Growth")