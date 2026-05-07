from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def extract_drug_reward(path, condition_suffix):
    # Extract the reward value from filenames such as:
    # EVAL_DRUG_R25_NO_GROWTH_Q_EP15k_TIME_...
    # Evaluation_Results_logbook_q_table_drug_reward_25_no_growth_EP_5000_TIME_...
    match = re.search(
        rf"(?:DRUG_R|drug_reward_)(\d+)_{re.escape(condition_suffix)}",
        path.stem,
        flags=re.IGNORECASE
    )
    return int(match.group(1)) if match else None


def find_evaluation_csvs(results_folder, condition_suffix):
    results_folder = Path(results_folder)
    csv_files = [
        csv_file
        for csv_file in results_folder.glob("*.csv")
        if extract_drug_reward(csv_file, condition_suffix) is not None
    ]

    return sorted(
        csv_files,
        key=lambda csv_file: extract_drug_reward(csv_file, condition_suffix)
    )


def load_evaluation_results(results_folder, condition_suffix):
    results_folder = Path(results_folder)
    csv_files = find_evaluation_csvs(results_folder, condition_suffix)

    if len(csv_files) == 0:
        raise FileNotFoundError(
            f"No drug evaluation CSV files with suffix '{condition_suffix}' found in: {results_folder}"
        )

    all_dfs = []
    for csv_file in csv_files:
        reward = extract_drug_reward(csv_file, condition_suffix)
        df = pd.read_csv(csv_file)
        df["Drug_Reward"] = reward
        all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True)


def plot_drug_reward_boxplots(results_folder, condition_suffix="no_growth", metrics=None, ylims=None):
    full_df = load_evaluation_results(results_folder, condition_suffix)
    rewards = sorted(full_df["Drug_Reward"].unique())
    labels = [f"Reward {r}" for r in rewards]
    if metrics is None:
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
            reward_data = full_df.loc[full_df["Drug_Reward"] == reward, metric].dropna()
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
        if ylims is not None and metric in ylims and ylims[metric] is not None:
            plt.ylim(ylims[metric])
        plt.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.show()


def plot_food_vs_drugs(results_folder, condition_suffix="no_growth", xlim=None, ylim=None):
    full_df = load_evaluation_results(results_folder, condition_suffix)
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
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_loop_death_rate(results_folder, condition_suffix="no_growth", ylim=None):
    full_df = load_evaluation_results(results_folder, condition_suffix)
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
    if ylim is not None:
        plt.ylim(ylim)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    results_folder = Path(__file__).resolve().parent / "Drugs_Energy_Yes_Penalty"
    condition_suffix = "ENG_PEN_FAC9"

    plot_drug_reward_boxplots(
        results_folder,
        condition_suffix=condition_suffix,
        ylims={
            "Drugs_Consumed": (0, 150),
            "Food_Consumed": (0, 150),
            "Snake_Length": (0, 100),
        }
    )
    plot_food_vs_drugs(results_folder, condition_suffix=condition_suffix)
    plot_loop_death_rate(results_folder, condition_suffix=condition_suffix, ylim=(0, 100))
