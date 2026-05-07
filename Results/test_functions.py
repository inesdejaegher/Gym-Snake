from pathlib import Path
from plot_functions import extract_drug_reward, find_evaluation_csvs, load_evaluation_results, plot_baseline_boxplots

results_path = Path("./Results")

plot_baseline_boxplots(results_path, subfolders=("Base", "Base_Energy"), metrics=None)