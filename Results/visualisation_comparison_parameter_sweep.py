from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_drug_reward_boxplots(results_folder):
    results_folder = Path(results_folder)

    def extract_reward(path):
        #Extract the reward value from the filename
        match = re.search(r"drug_reward_(\d+)_growth", path.stem)
        return int(match.group(1)) if match else None

    # Get and sort CSV files
    csv_files = [
        csv_file
        for csv_file in results_folder.glob("*.csv")
        if extract_reward(csv_file) is not None
    ]
    csv_files = sorted(csv_files, key=extract_reward)

    if len(csv_files) == 0:
        raise FileNotFoundError(f"No drug with-growth evaluation CSV files found in: {results_folder}")

    rewards = sorted({extract_reward(f) for f in csv_files})
    labels = [f"Reward {r}" for r in rewards]

    # Load data and group repeated evaluations of the same reward together.
    dfs_by_reward = {
        reward: [
            pd.read_csv(csv_file)
            for csv_file in csv_files
            if extract_reward(csv_file) == reward
        ]
        for reward in rewards
    }

    metrics = [
        "Drugs_Consumed",
        "Food_Consumed",
        "Total_Reward",
        "Snake_Length"
    ]

    # Create one figure per metric
    for metric in metrics:
        data = []
        for reward in rewards:
            reward_data = pd.concat(
                [df[metric] for df in dfs_by_reward[reward]],
                ignore_index=True
            ).dropna()
            data.append(reward_data)

        plt.figure(figsize=(12, 5))
        plt.boxplot(data, tick_labels=labels, showmeans=True)
        
        # Compute means for each reward
        means = [d.mean() for d in data]

        # X positions (1-based because boxplot uses 1..N)
        x = np.arange(1, len(means) + 1)

        # Plot trendline
        plt.plot(x, means, linestyle='--', marker='o', linewidth=2, label='Mean trend')

        plt.legend()
        plt.title(f"{metric} per game by Drug Reward")
        plt.xlabel("Drug Reward")
        plt.ylabel(metric)
        plt.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.show()


def plot_food_vs_drugs(results_folder):
    results_folder = Path(results_folder)

    def extract_reward(path):
        match = re.search(r"drug_reward_(\d+)_growth", path.stem)
        return int(match.group(1)) if match else None

    csv_files = [
        f for f in results_folder.glob("*.csv")
        if extract_reward(f) is not None
    ]

    all_dfs = []

    for csv_file in csv_files:
        reward = extract_reward(csv_file)
        df = pd.read_csv(csv_file)
        df["Drug_Reward"] = reward
        all_dfs.append(df)

    full_df = pd.concat(all_dfs, ignore_index=True)

    summary = (
        full_df
        .groupby("Drug_Reward")
        .agg(
            mean_food=("Food_Consumed", "mean"),
            mean_drugs=("Drugs_Consumed", "mean"),
        )
        .reset_index()
        .sort_values("Drug_Reward")
    )

    plt.figure(figsize=(7, 5))

    # scatter only
    plt.scatter(summary["mean_food"], summary["mean_drugs"])

    # label each point
    for _, row in summary.iterrows():
        plt.text(
            row["mean_food"],
            row["mean_drugs"],
            f"R{int(row['Drug_Reward'])}",
            fontsize=9,
            ha="left",
            va="bottom"
        )
    sorted_summary = summary.sort_values("mean_food")

    plt.plot(
        sorted_summary["mean_food"],
        sorted_summary["mean_drugs"],
        "o--",
        color="orange",
        linewidth=2,
        markersize=6
)
    plt.xlabel("Mean Food Consumed")
    plt.ylabel("Mean Drugs Consumed")
    plt.title("Food–Drug Trade-off (Means Only)")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_loop_death_rate(results_folder):
    results_folder = Path(results_folder)

    def extract_reward(path):
        match = re.search(r"drug_reward_(\d+)_growth", path.stem)
        return int(match.group(1)) if match else None

    csv_files = [
        f for f in results_folder.glob("*.csv")
        if extract_reward(f) is not None
    ]

    all_dfs = []

    for csv_file in csv_files:
        reward = extract_reward(csv_file)
        df = pd.read_csv(csv_file)
        df["Drug_Reward"] = reward
        all_dfs.append(df)

    full_df = pd.concat(all_dfs, ignore_index=True)

    summary = (
        full_df
        .groupby("Drug_Reward")
        .agg(
            loop_deaths=("Loop", "sum"),
            total_evals=("Loop", "count"),
        )
        .reset_index()
        .sort_values("Drug_Reward")
    )

    summary["loop_death_rate"] = 100 * summary["loop_deaths"] / summary["total_evals"]

    plt.figure(figsize=(8, 5))

    plt.bar(
        summary["Drug_Reward"].astype(str),
        summary["loop_death_rate"]
    )

    for _, row in summary.iterrows():
        plt.text(
            str(int(row["Drug_Reward"])),
            row["loop_death_rate"],
            f"{int(row['loop_deaths'])}/{int(row['total_evals'])}",
            ha="center",
            va="bottom"
        )

    plt.xlabel("Drug Reward")
    plt.ylabel("Loop Death Rate (%)")
    plt.title("Percentage of Evaluations Ending in Loop Death")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_drug_reward_boxplots(Path(__file__).resolve().parent / "Drugs_With_Growth")
    plot_food_vs_drugs(Path(__file__).resolve().parent / "Drugs_With_Growth")
    plot_loop_death_rate(Path(__file__).resolve().parent / "Drugs_With_Growth")