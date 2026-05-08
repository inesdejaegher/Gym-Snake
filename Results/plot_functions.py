from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----- LOADING -----
def extract_drug_reward(path, condition_suffix=""):
    """
    The extract_drug_reward function is designed to parse the name of a file (provided as a pathlib.Path object) 
    and extract the numeric "drug reward" value associated with that file's experiment.

    "EVAL_DRUG_R25_NO_GROWTH_Q_EP15k_TIME_XXX"
    Returns --> 25

    Input:
    - path = Path to the CSV file
    - condition_suffix = 
    Output:
    - reward_value
    """
    if condition_suffix:
        pattern = rf"(?:DRUG_R|drug_reward_)(\d+)_{re.escape(condition_suffix)}"
    else:
        pattern = r"(?:DRUG_R|drug_reward_)(\d+)"
        
    match = re.search(pattern, path.stem, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None

def load_evaluation_results(results_folder, condition_suffix):
    results_folder = Path(results_folder)
    # Gather list of csv files
    csv_files = [csv_file for csv_file in results_folder.glob("*.csv")
        if extract_drug_reward(Path(csv_file), condition_suffix) is not None]
    
    csv_files_list = sorted(csv_files, key=lambda csv_file: extract_drug_reward(csv_file, condition_suffix))

    if len(csv_files_list) == 0:
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

# -----------------------------
# ----- PLOTTING BASELINE -----
# -----------------------------
def plot_baseline_boxplots(general_results_folder, subfolders=("Base", "Base_Energy")):
    """
    Plots boxplots for the metrics found in the CSV files of the specified subfolders.
    
    Input:
    - general_results_folder = Path to the main results directory
    - subfolders = Tuple or list of folder names to compare (e.g., "Base" vs "Base_Energy")
    - metrics = List of metrics (columns) to plot
    """
    general_results_folder = Path(general_results_folder)
    metrics = ["Food_Consumed", "Total_Reward", "Snake_Length", "Steps"]
        
    # Load the CSV data from the specified folders
    data_dict = {}
    for folder_name in subfolders:
        folder_path = general_results_folder / folder_name
        folder_path = Path(folder_path)
        if not folder_path.exists():
            print(f"Warning: Folder '{folder_path}' does not exist.")
            continue
            
        csv_files = list(folder_path.glob("*.csv"))
        if not csv_files:
            print(f"Warning: No CSV files found in '{folder_path}'.")
            continue
            
        # Read and concatenate all CSVs in this subfolder
        dfs = [pd.read_csv(f) for f in csv_files]

        # data_dict will hold the dataframes we have from the baseline runs in a dictionary 
        data_dict[folder_name] = pd.concat(dfs, ignore_index=True)
        
    if not data_dict:
        print("No data found to plot.")
        return
        
    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        plot_data = []
        labels = []
        
        for folder_name in subfolders:
            if folder_name in data_dict and metric in data_dict[folder_name].columns:
                # Extract the metric data and drop NaNs just in case
                metric_data = data_dict[folder_name][metric].dropna()
                plot_data.append(metric_data)
                labels.append(folder_name)
                
        if not plot_data:
            print(f"Warning: No data found for metric '{metric}'.")
            ax.set_visible(False)
            continue
            
        ax.boxplot(
            plot_data,
            labels=labels,
            showmeans=True,
            meanprops={
                "marker": "s",
                "markerfacecolor": "green",
                "markeredgecolor": "green",
            },
        )
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        # Compute means for each condition
        means = [d.mean() for d in plot_data]
        x = np.arange(1, len(means) + 1)
        ax.plot(x, means, linestyle='--', marker='o', color='red', linewidth=2, label='Mean trend', alpha=0.5)
        for x_pos, mean_value in zip(x, means):
            ax.annotate(
                f"{mean_value:.1f}",
                xy=(x_pos, mean_value),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=11,
                color="green",
            )
        
        ax.legend(fontsize=12)
        ax.set_title(f"{metric} Comparison: {' vs '.join(labels)}", fontsize=16)
        ax.set_ylabel(metric, fontsize=14)
        ax.grid(axis="y", alpha=0.3)
        
    plt.tight_layout()
    plt.show()

def plot_baseline_boxplots_metrics(general_results_folder, subfolders=("Base", "Base_Energy"), metrics=None):
    """
    Plots boxplots for the metrics found in the CSV files of the specified subfolders.
    
    Input:
    - general_results_folder = Path to the main results directory
    - subfolders = Tuple or list of folder names to compare (e.g., "Base" vs "Base_Energy")
    - metrics = List of metrics (columns) to plot
    """
    general_results_folder = Path(general_results_folder)
    
    if metrics is None:
        metrics = [
            "Food_Consumed",
            "Total_Reward",
            "Snake_Length",
            "Steps"
        ]
        
    # Load the CSV data from the specified folders
    data_dict = {}
    for folder_name in subfolders:
        folder_path = general_results_folder / folder_name
        folder_path = Path(folder_path)
        if not folder_path.exists():
            print(f"Warning: Folder '{folder_path}' does not exist.")
            continue
            
        csv_files = list(folder_path.glob("*.csv"))
        if not csv_files:
            print(f"Warning: No CSV files found in '{folder_path}'.")
            continue
            
        # Read and concatenate all CSVs in this subfolder
        dfs = [pd.read_csv(f) for f in csv_files]

        # data_dict will hold the dataframes we have from the baseline runs in a dictionary 
        data_dict[folder_name] = pd.concat(dfs, ignore_index=True)
        
    if not data_dict:
        print("No data found to plot.")
        return
        
    # Create one figure per metric
    for metric in metrics:
        plt.figure(figsize=(8, 5))
        plot_data = []
        labels = []
        
        for folder_name in subfolders:
            if folder_name in data_dict and metric in data_dict[folder_name].columns:
                # Extract the metric data and drop NaNs just in case
                metric_data = data_dict[folder_name][metric].dropna()
                plot_data.append(metric_data)
                labels.append(folder_name)
                
        if not plot_data:
            print(f"Warning: No data found for metric '{metric}'.")
            continue
            
        plt.boxplot(plot_data, labels=labels, showmeans=True)
        
        # Compute means for each condition
        means = [d.mean() for d in plot_data]
        x = np.arange(1, len(means) + 1)
        plt.plot(x, means, linestyle='--', marker='o', color='red', linewidth=2, label='Mean trend')
        plt.tick_params(axis='both', which='major', labelsize=14)
        
        plt.legend(fontsize=11)
        plt.title(f"{metric} Comparison: {' vs '.join(labels)}",fontsize=16)
        plt.ylabel(metric, fontsize=14)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()

# -------------------------------
# ----- PLOTTING CONDITIONS -----
# -------------------------------
def plot_condition_boxplots(results_folder, condition_suffix, metrics=None, ylims=None):
    """
    Reads multiple CSV files, extracts the condition value (Drug Reward), 
    concatenates them into a single DataFrame, and plots side-by-side 
    boxplots for given metrics separated by the condition.
    """
    # Reuse the existing function to load, extract condition, add column, and concatenate!
    full_df = load_evaluation_results(results_folder, condition_suffix)
    
    if metrics is None:
        metrics = [
            "Food_Consumed",
            "Drugs_Consumed",
            "Total_Reward",
            "Snake_Length",
            "Steps"
        ]
        
    condition_col = "Drug_Reward"
    conditions = sorted(full_df[condition_col].unique())
    labels = [f"Reward {cond}" for cond in conditions]
    
    for metric in metrics:
        if metric not in full_df.columns:
            print(f"Warning: Metric '{metric}' not found in data.")
            continue
            
        plt.figure(figsize=(10, 6))
        plot_data = []
        
        for cond in conditions:
            metric_data = full_df.loc[full_df[condition_col] == cond, metric].dropna()
            plot_data.append(metric_data)
            
        plt.boxplot(plot_data, labels=labels, showmeans=True)
        
        means = [d.mean() for d in plot_data]

        x = np.arange(1, len(means) + 1)
        plt.plot(x, means, linestyle='--', marker='o', color='red', linewidth=2, label='Mean trend')

        if ylims is not None:
            plt.ylim(ylims)
        
        plt.title(f"{metric} Comparison by {condition_col}")
        plt.xlabel("Condition")
        plt.ylabel(metric)
        plt.legend()
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()

def plot_condition_boxplots_grid(results_folder, condition_suffix, metrics=None, ylims=None):
    """
    Reads multiple CSV files, extracts the Drug Reward condition, and plots
    multiple metric boxplots together in one 2x2 figure.
    """
    full_df = load_evaluation_results(results_folder, condition_suffix)

    if metrics is None:
        metrics = [
            "Food_Consumed",
            "Drugs_Consumed",
            "Snake_Length",
            "Steps",
        ]

    condition_col = "Drug_Reward"
    conditions = sorted(full_df[condition_col].unique())
    labels = [f"Reward {cond}" for cond in conditions]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        if metric not in full_df.columns:
            print(f"Warning: Metric '{metric}' not found in data.")
            ax.set_visible(False)
            continue

        plot_data = []
        for cond in conditions:
            metric_data = full_df.loc[full_df[condition_col] == cond, metric].dropna()
            plot_data.append(metric_data)

        ax.boxplot(
            plot_data,
            labels=labels,
            showmeans=True,
            meanprops={
                "marker": "s",
                "markerfacecolor": "green",
                "markeredgecolor": "green",
            },
        )

        means = [d.mean() for d in plot_data]
        x = np.arange(1, len(means) + 1)
        ax.plot(x, means, linestyle="--", marker="o", color="red", linewidth=2, label="Mean trend")


        if ylims is not None and metric in ylims:
            ax.set_ylim(ylims[metric])

        ax.set_title(f"{metric} by {condition_col}", fontsize=14)
        ax.set_xlabel("Condition")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)
        ax.legend()

    for ax in axes[len(metrics):]:
        ax.set_visible(False)

    fig.suptitle(f"Condition Comparison: {condition_suffix}", fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_preference_ratio_by_reward(results_folder, condition_suffix, ylim=None):
    """
    Plots the mean Preference_Ratio for each drug reward condition.
    Infinite ratios are ignored in the mean because they occur when Food_Consumed is 0.
    """
    full_df = load_evaluation_results(results_folder, condition_suffix)

    if "Preference_Ratio" not in full_df.columns:
        if {"Drugs_Consumed", "Food_Consumed"}.issubset(full_df.columns):
            full_df["Preference_Ratio"] = full_df["Drugs_Consumed"] / full_df["Food_Consumed"]
        else:
            raise ValueError("Preference_Ratio could not be found or computed from the data.")

    ratio_df = full_df.copy()
    ratio_df["Preference_Ratio"] = ratio_df["Preference_Ratio"].replace([np.inf, -np.inf], np.nan)

    summary = (
        ratio_df
        .groupby("Drug_Reward")["Preference_Ratio"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("Drug_Reward")
    )
    summary["sem"] = summary["std"] / np.sqrt(summary["count"])

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        summary["Drug_Reward"],
        summary["mean"],
        yerr=summary["sem"],
        marker="o",
        linewidth=2,
        capsize=4,
        color="purple",
        label="Mean preference ratio",
    )

    for _, row in summary.iterrows():
        plt.annotate(
            f"{row['mean']:.1f}",
            xy=(row["Drug_Reward"], row["mean"]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    if ylim is not None:
        plt.ylim(ylim)

    plt.title(f"Preference Ratio by Drug Reward: {condition_suffix}")
    plt.xlabel("Drug Reward")
    plt.ylabel("Preference Ratio (Drugs / Food)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_penalty_vs_no_penalty_metric(
    no_penalty_folder,
    no_penalty_suffix,
    penalty_folder,
    penalty_suffix,
    metric="Drugs_Consumed",
    ylim=None,
):
    """
    Compares one metric across matching drug reward levels for the no-penalty
    and penalty conditions.
    """
    no_penalty_df = load_evaluation_results(no_penalty_folder, no_penalty_suffix)
    penalty_df = load_evaluation_results(penalty_folder, penalty_suffix)

    if metric not in no_penalty_df.columns:
        raise ValueError(f"Metric '{metric}' not found in no-penalty data.")
    if metric not in penalty_df.columns:
        raise ValueError(f"Metric '{metric}' not found in penalty data.")

    def summarize(df):
        summary = (
            df
            .groupby("Drug_Reward")[metric]
            .agg(["mean", "std", "count"])
            .reset_index()
            .sort_values("Drug_Reward")
        )
        summary["sem"] = summary["std"] / np.sqrt(summary["count"])
        return summary

    no_penalty_summary = summarize(no_penalty_df)
    penalty_summary = summarize(penalty_df)

    common_rewards = sorted(
        set(no_penalty_summary["Drug_Reward"]).intersection(penalty_summary["Drug_Reward"])
    )
    no_penalty_summary = no_penalty_summary[no_penalty_summary["Drug_Reward"].isin(common_rewards)]
    penalty_summary = penalty_summary[penalty_summary["Drug_Reward"].isin(common_rewards)]

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        no_penalty_summary["Drug_Reward"],
        no_penalty_summary["mean"],
        yerr=no_penalty_summary["sem"],
        marker="o",
        linewidth=2,
        capsize=4,
        label="No penalty",
    )
    plt.errorbar(
        penalty_summary["Drug_Reward"],
        penalty_summary["mean"],
        yerr=penalty_summary["sem"],
        marker="s",
        linewidth=2,
        capsize=4,
        label="Penalty",
    )

    if ylim is not None:
        plt.ylim(ylim)

    plt.title(f"{metric}: No Penalty vs Penalty")
    plt.xlabel("Drug Reward")
    plt.ylabel(metric)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
