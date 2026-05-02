# Import needed modules
import numpy as np
np.bool8 = np.bool_
import logging
import os
import warnings
warnings.filterwarnings("ignore")

from helper_func import plot_preference_ratio_from_csv, plot_reward_from_csv, plot_epsilon_from_csv, plot_snake_length_from_csv


# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

folder = os.path.join(os.path.dirname(__file__), "..", "Results")


# -------------------------------------
# ----- PLOTTING BASELINE RESULTS -----
# -------------------------------------
CSV_NAME = "FOOD-DRUG_preference_logbook_TIME_02_05_2026_12-00-58.csv"
full_csv_path = os.path.join(folder, CSV_NAME) # Combine folder and file name

# ----- PLOT PREFERENCE RATIO -----
plot_preference_ratio_from_csv(full_csv_path)

# ----- PLOT TOTAL REWARD -----
plot_reward_from_csv(full_csv_path)

# ----- PLOT EPSILON DECAY -----
plot_epsilon_from_csv(full_csv_path)

# ----- PLOT SNAKE LENGTH -----
plot_snake_length_from_csv(full_csv_path)
