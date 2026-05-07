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
            
        ax.boxplot(plot_data, labels=labels, showmeans=True)
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        # Compute means for each condition
        means = [d.mean() for d in plot_data]
        x = np.arange(1, len(means) + 1)
        ax.plot(x, means, linestyle='--', marker='o', color='red', linewidth=2, label='Mean trend', alpha=0.5)
        
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
